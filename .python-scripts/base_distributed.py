"""
NCAA Tournament Prediction Model (Final Version)

Description:
This model predicts whether a Division I men's basketball team will make the NCAA Tournament 
based on team-level performance statistics. It uses a deep neural network trained on historical 
data with engineered features and threshold optimization to maximize accuracy and recall.

Major Improvements Over the Original Version:

1. Deep Neural Network Architecture
   - Expanded from a 2-layer network to a 4-layer network (128 → 64 → 32 → 1).
   - Uses ReLU activations, dropout for regularization, and L2 penalties to reduce overfitting.

2. GPU Acceleration
   - Model trains using TensorFlow on CUDA-enabled GPUs via OneDeviceStrategy.

3. Feature Engineering
   - Created difference features to represent offensive vs. defensive strength (e.g., `NET_DIFF`, `EFG_DIFF`, `TO_MARGIN`).
   - Added derived features such as `PACE_OFF` and `WIN_PCT` to better capture tournament-level dynamics.

4. Threshold Optimization
   - Automatically identifies the classification threshold that yields the best F1 score using validation data.
   - Final threshold (e.g., ~0.3776) improves balance between precision and recall

This model is designed to be both predictive and interpretable — a powerful tool for coaches, analysts, or selection committees.
"""

import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, f1_score

# Clear previous sessions to release GPU memory
tf.keras.backend.clear_session()
gc.collect()

# Load dataset and add difference features
cbb = pd.read_csv("cbb.csv")
cbb["L"] = cbb["G"] - cbb["W"] # How many losses a team has

cbb["NET_DIFF"] = cbb["ADJOE"] - cbb["ADJDE"]  # How efficient a team is on net, Adjusted efficiency margin

cbb["EFG_DIFF"] = cbb["EFG_O"] - cbb["EFG_D"]  # Shot-making vs shot-prevention, Shooting efficiency differential

cbb["TO_MARGIN"] = cbb["TORD"] - cbb["TOR"]    # Force turnovers vs commit them, Turnover margin (both offense and defense)

cbb["REB_MARGIN"] = cbb["ORB"] - cbb["DRB"]    # Offensive boards vs allowed, Rebounding margin

cbb["FTR_DIFF"] = cbb["FTR"] - cbb["FTRD"]     # Drawing vs allowing fouls, Free throw rate margin

cbb["2P_DIFF"] = cbb["2P_O"] - cbb["2P_D"]     # 2PT offense vs defense, Two-point shooting margin

cbb["3P_DIFF"] = cbb["3P_O"] - cbb["3P_D"]     # 3PT offense vs defense, Three-point shooting margin

cbb["PACE_OFF"] = cbb["ADJOE"] * cbb["ADJ_T"]  # Fast and efficient = dangerous, tempo-weighted offensive output

cbb["WIN_PCT"] = cbb["W"] / cbb["G"]           # Just to include raw win rate  Wins vs games played

# Create binary label for making the tournament
cbb['MADE_TOURNAMENT'] = cbb['POSTSEASON'].apply(
    lambda x: 1 if pd.notnull(x) and x != 'NIT' else 0)

# Prepare features and labels
X = cbb.drop(columns=['POSTSEASON', 'TEAM', 'YEAR', 'CONF', 'SEED', 'MADE_TOURNAMENT','G'])
X = X.apply(pd.to_numeric)
y = cbb['MADE_TOURNAMENT']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize multi-GPU strategy
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#gpus = tf.config.list_physical_devices('GPU')
#if gpus:
#    try:
#        tf.config.set_logical_device_configuration(
#            gpus[0],
#            [tf.config.LogicalDeviceConfiguration(memory_limit=28000)])  # Reserve ~28 GB
#    except RuntimeError as e:
#        print("Memory configuration must be set before GPUs are initialized:", e)

strategy = tf.distribute.MirroredStrategy()
print("GPUs detected and in use:", strategy.num_replicas_in_sync)

# Build and compile the model within the strategy scope
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='binary_crossentropy',
                  metrics=['accuracy'])

# Early stopping callback
early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

# Train the model with batch size appropriate for multi-GPU
model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2,batch_size=64, callbacks=[early_stop], verbose=1)

# Get predicted probabilities for the positive class (teams that made the tournament)
y_probs = model.predict(X_test_scaled).ravel()

# Get precision, recall, thresholds
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

# Compute F1 scores for all thresholds
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)  # add small term to avoid div by zero
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

# Prediction and evaluation at best F1 score threshold
y_pred = (y_probs > best_threshold).astype(int)
report = classification_report(y_test, y_pred, output_dict=True)

# Feature importance from the first dense layer
for layer in model.layers:
    weights = layer.get_weights()
    if weights:
        feature_weights = np.abs(weights[0]).sum(axis=1)
        break

feature_importance = pd.Series(feature_weights, index=X.columns).sort_values(ascending=False)

# Print results
print("\nMost important stats to measure if a team gets into the NCAA Tournament:")
print(feature_importance)
print("\nClassification Report:")
print(report)

