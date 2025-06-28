#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 09:41:59 2025

@author: suhaimi.sulaiman
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score

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

# Train KNN model
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
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

# ROC Curve and AUC Score
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, marker='.', linewidth=2, label=f'KNN (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - KNN Classifier')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"ROC AUC Score: {auc_score:.3f}")

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
plot_decision_boundary(X_train, y_train, classifier, 'KNN Classifier (Training set)', xlabel='Age', ylabel='Estimated Salary')
plot_decision_boundary(X_test, y_test, classifier, 'KNN Classifier (Test set)', xlabel='Age', ylabel='Estimated Salary')

