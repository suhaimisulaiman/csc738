#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 16:16:09 2025

@author: suhaimi.sulaiman
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load dataset
dataset = pd.read_csv('../data/raw/LR.csv')
X = dataset.iloc[:, [0, 1]].values  # Features: Age and Salary
y = dataset.iloc[:, 2].values      # Target: Purchased

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train logistic regression model
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluation
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display evaluation
print("Confusion Matrix:\n", cm)
print("\nAccuracy Score:", accuracy)
print("\nClassification Report:\n", report)

# Define reusable plot function
def plot_decision_boundary(X, y, classifier, title, xlabel='Feature 1', ylabel='Feature 2'):
    X1, X2 = np.meshgrid(
        np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01),
        np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)
    )
    plt.contourf(
        X1, X2,
        classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
        alpha=0.75, cmap=ListedColormap(('red', 'green'))
    )
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y)):
        plt.scatter(
            X[y == j, 0], X[y == j, 1],
            color=ListedColormap(('red', 'green'))(i), label=j
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

# Visualize training and test results
plot_decision_boundary(X_train, y_train, classifier, 'Logistic Regression (Training set)', xlabel='Age', ylabel='Estimated Salary')
plot_decision_boundary(X_test, y_test, classifier, 'Logistic Regression (Test set)', xlabel='Age', ylabel='Estimated Salary')
