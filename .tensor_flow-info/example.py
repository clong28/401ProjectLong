import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load dataset
cbb = pd.read_csv("cbb.csv")

cbb['MADE_TOURNAMENT'] = cbb['POSTSEASON'].apply(
    lambda x: 1 if pd.notnull(x) and x != 'NIT' else 0
)

# Drop columns that shouldn't be used as features
X = cbb.drop(columns=['POSTSEASON', 'TEAM', 'YEAR', 'CONF','SEED','MADE_TOURNAMENT'])
X = X.apply(pd.to_numeric)
y = cbb['MADE_TOURNAMENT']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a simple neural network model
model = tf.keras.Sequential([

    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),

    tf.keras.layers.Dense(16, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])

# Compile the model
model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, validation_split=0.2, verbose=0)

# Predict and evaluate
y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
report = classification_report(y_test, y_pred, output_dict=True)

# Feature importance from first dense layer weights

feature_weights = np.abs(model.layers[0].get_weights()[0]).sum(axis=1)
feature_importance = pd.Series(feature_weights, index=X.columns).sort_values(ascending=False)
print("Most important stats to measure if a team gets into the NCAA Tournament",feature_importance)
print(report)
