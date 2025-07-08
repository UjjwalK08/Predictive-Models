# 🧠 Machine Learning Models

Welcome to the **Predictive Models** repository! This collection features diverse machine learning projects, each focused on solving real-world classification or regression problems using popular algorithms. All datasets are publicly available from trusted sources like the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).

---

## 📌 Projects Overview

### 1. 🩺 Breast Cancer Detection

**Data:** [Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data) | [Description](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names)

**Overview:**
This project predicts breast cancer diagnoses (malignant vs. benign) using machine learning models including:

* Decision Tree
* K-Nearest Neighbors (KNN)
* Logistic Regression
* Support Vector Machine (SVM)

The dataset contains real-valued features representing cell nucleus characteristics (mean, standard error, and worst-case values). Feature engineering and preprocessing were applied prior to training.

---

### 2. 🚗 Car Evaluation Classification

**Data:** [Dataset](http://archive.ics.uci.edu/ml/datasets/Car+Evaluation)

**Overview:**
Implemented multiple classifiers to predict the evaluation category of a car (unacceptable, acceptable, good, very good) based on:

* Buying price
* Maintenance cost
* Number of doors
* Person capacity
* Luggage boot size
* Estimated safety

Algorithms used include:

* Decision Tree
* KNN
* Logistic Regression
* SVM
* Naive Bayes

Categorical features were preprocessed, and the target variable was label-encoded for training.

---

### 3. 💳 Predictive Modeling of Consumer Spending

**Problem Statement:**
Built and compared several regression models to forecast consumer spending after catalog mailings. Algorithms included:

* Linear Regression
* k-Nearest Neighbors
* Regression Trees
* Support Vector Regression
* Neural Networks
* Ensemble Methods

Two approaches were tested:

* Using the full dataset (all customers)
* Using only data from purchasers

Hyperparameter tuning and normalization were performed to maximize predictive accuracy.

---

### 4. 📬 Cost-Sensitive Spam Email Detection

**Data:** [Spambase Dataset](https://archive.ics.uci.edu/dataset/94/spambase)

**Problem Statement:**
Developed two types of classification models for spam detection:

* **Accuracy-Optimized Model:** Maximized standard performance metrics
* **Cost-Sensitive Model:** Minimized misclassification costs with a **10:1 penalty** for false negatives over false positives

Techniques applied:

* k-NN
* Decision Trees
* Naive Bayes
* SVM
* Ensemble Methods

Models were evaluated using:

* Nested Cross-Validation
* Confusion Matrices
* ROC Curves
* Lift Charts
* Metrics: Accuracy, F1-score, AUC, and Average Cost

---

### 5. 🔬 Shallow vs. Deep Neural Networks for Function Approximation

**Problem Statement:**
Compared neural networks with 1, 2, and 3 hidden layers to approximate the non-linear function:

```math
f(x) = 2(2\cos^2(x) - 1)^2 - 1
```

**Experiment Setup:**

* Generated 120,000 points sampled uniformly from $[-2\pi, 2\pi]$
* Split into training and test sets
* Trained networks with varying hidden units
* Evaluation Metric: Mean Squared Error (MSE)

**Key Insight:**
Deeper architectures (2–3 layers) achieved significantly **lower errors with fewer parameters**, demonstrating superior ability to capture complex functions.

![Function Approximation Results](https://github.com/user-attachments/assets/3db37477-84e6-4a6a-9da6-2516a3d64cf5)

---

## 🚀 Getting Started

Each project folder contains:

* Python notebooks/scripts
* Data loading and preprocessing steps
* Model training, tuning, and evaluation
* Visualization of results and performance metrics

To run any project:

1. Clone this repository
2. Navigate to the desired project directory
3. Install dependencies listed in `requirements.txt` (if provided)
4. Run the notebook or script

---

## 📬 Contact

Feel free to connect or reach out for collaboration or questions!
[LinkedIn](https://www.linkedin.com/in/ujjwalkhanna15) | [GitHub](https://github.com/)

