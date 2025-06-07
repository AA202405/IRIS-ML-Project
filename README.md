# 🌸 Iris Flower Classification - Machine Learning Project

This project demonstrates a basic end-to-end **machine learning workflow** using Python and the Iris dataset. It performs data loading, visualization, model training, and evaluation using several classification algorithms.

---

## 📁 Dataset
The **Iris dataset** is a classic dataset in pattern recognition and machine learning, consisting of 150 samples from three species of Iris flowers — *setosa*, *versicolor*, and *virginica*.

- **Features**:
  - Sepal length
  - Sepal width
  - Petal length
  - Petal width
- **Target**:
  - Iris-setosa
  - Iris-versicolor
  - Iris-virginica

Dataset source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)

---

## 🧠 Algorithms Used
We spot-checked the following classification algorithms using **10-fold cross-validation**:
- Logistic Regression (LR)
- Linear Discriminant Analysis (LDA)
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier (CART)
- Gaussian Naive Bayes (NB)
- Support Vector Machine (SVM)

---

## 📊 Project Structure
- **Data Loading** using Pandas
- **Exploratory Data Analysis (EDA)**: boxplots, histograms, scatter plots
- **Model Training** using Scikit-learn
- **Model Evaluation** using accuracy and standard deviation via cross-validation

---

## 🛠 Requirements

Install the required packages using pip:

```bash
pip install numpy pandas matplotlib scikit-learn
