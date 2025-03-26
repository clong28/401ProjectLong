# Big Data, Bigger Upsets: Letting AI Embrace the Madness
By Connor Long

---
<img alt="Simple Icon of a camera. This is just a place holder for your image" src="https://media.licdn.com/dms/image/D4D12AQH5Hnkq-G2Ucw/article-cover_image-shrink_720_1280/0/1691263849733?e=2147483647&v=beta&t=eYdGi07vRtKUnO1mPGi46KVE6UnaB_x3A767DwpF__s" width="75%">
Image from: https://media.licdn.com/dms/image/D4D12AQH5Hnkq-G2Ucw/article-cover_image-shrink_720_1280/0/1691263849733?e=2147483647&v=beta&t=eYdGi07vRtKUnO1mPGi46KVE6UnaB_x3A767DwpF__s

---
### Abstract
*For my project, I chose to look at college basketball, specifically focusing on the NCAA tournament. The motivation behind this choice stems from the tournament’s unpredictabability and popularity, along with it being current with MSU involved to see how accurate I really was. Through statistical modeling, machine learning, and data analysis I will try to find features that will help me predict each matchup. I plan to explore predictive modeling software using Python libraries like scikit-learn and deep learning tools like TensorFlow. Hardware-wise, I’ll be running experiments on a high-performance computing cluster with a GPU in case I need to greatly speed up the process. I'd want to benchmark different model types and feature sets to see which has the most accurate predictions. A successful outcome would be a model that can consistently identify and predict first and second round matchups with accuracy. Along with predicting the tournament winner or Final Four with greater accuracy than common baselines. Common baselines is seeding, AKA the higher seed wins.*

----
### Schedule

* **Sunday February 5** - *Project Proposal Milestone Due*
* **Sunday March 30** - *Project Part 1 Due*
* **March 23-29:** - *Finalize data cleaning and basic feature selection. Wrap up preprocessing logic. Ensure all Part 1 code and markdowns are polished.*
* **Mar 30** - *Submit Part 1.*
* **April 1-7** - *Construct matchup-based training data using historical stats, making features like seed difference and existing feature deltas. Build a basic TensorFlow model to predict matchup outcomes. Model will be trained and evaluated with simple metrics like accuracy and AUC to get a good baseline.*
* **April 8-14** - *Focus on improving your model this week by tuning hyperparameters, adding more sophisticated features like recent performance, number of hidden layers, different train/test splits and any others that might come up. Use tools to help with interpretation of model. Begin simulating the 2025 NCAA bracket using trained model and the stats from this year.*
* **April 15-21:** - *Wrap up the project, look over notebook and any files for submission, compare different model versions, recording the whole process, and adding visualizations and insights from my tournament simulations.*
* **IDK** - *Final Project due*

---
### Part 1 Software Exploration

*For the first part of this assignment, I will be reviewing software tools and frameworks relevant to building a predictive model for the NCAA Men’s Basketball Tournament. The primary software I will be investigating is TensorFlow, an open-source machine learning library developed by Google, which provides the core functionality for building, training, and evaluating neural networks which will be great for building an accurate model(https://www.tensorflow.org/). I will also be using Pandas and NumPy for data manipulation and preprocessing, and Matplotlib for initial data visualization which I have extensive experience with through my previouse MSU Data Science curriculum. I will revisit and explore scikit-learn for feature selection and model comparison which was introduced to me last year. I will start with analyzing prior tournaments, to identify key predictive features. I hope to clean and filter the datasets for relevant information within them to make a feature-rich dataset my models training. I hope to include a set of reproducible instructions for running the code on the HPCC using SLURM job scripts within a README file to guide any other user who is curious about further development.*

---
### Part 2 Benchmark and Optimization

*I plan to benchmark training a neural network using TensorFlow. TensorFlow supports parallel execution across CPUs and GPUs, and can be run on the HPCC with parallelization using OpenMP or GPU acceleration. I will focus on the model training, data loading and memory usage. Will see these scale when executed on different hardware. Some scaling studies are CPU vs GPU, single node vs multiple node. Can investigate further when running into complications. Success will be measured both by performance gains like a 3-5x speedup in training CPU vs GPU. Also by predictive accuracy of the resulting model (what my model predicted vs what actually happened). In the final report, I aim to deliver a reproducible benchmark analysis, working HPCC scripts, and an optimized version of the training code. Ideally, this will include performance plots and recommendations for further improvements. A successful outcome would be a speedup in model training with implemented efficiency without loss of model accuracy.*
