#!/usr/bin/env python
# coding: utf-8

import warnings

import numpy as np
import pandas as pd
from sklearn import preprocessing

# Gradient Boosting Classifier
# AdaBoost Classifier
# Random Forest
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score as f1_score_rep
from sklearn.model_selection import train_test_split

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

# k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

# Multi-layer Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

# Support Vector Machine
from sklearn.svm import SVC

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

# XGBoost
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

df = pd.read_csv("../data/preprocessed_data.csv", low_memory=False)


y = df["class1"]  # get the most detailed class labels (class1 > class2 > class3)
X = df.drop(columns=["class1", "class2", "class3"])
del df


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)
del X, y


scaler = preprocessing.MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


x_train = np.copy(X_train_scaled)
x_test = np.copy(X_test_scaled)


label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

clf = DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
dt_pred = clf.predict(x_test)


print("****************** Decision Tree prediction results ******************")
print("Accuracy: ", accuracy_score(y_test, dt_pred))
print("Micro F1 Score: ", f1_score_rep(y_test, dt_pred, average="micro"))
print("Macro F1 Score: ", f1_score_rep(y_test, dt_pred, average="macro"))


gb = GaussianNB()
gb = gb.fit(x_train, y_train)
gb_pred = gb.predict(x_test)


print("****************** Gaussian NB prediction results ******************")
print("Accuracy: ", accuracy_score(y_test, gb_pred))
print("Micro F1 Score: ", f1_score_rep(y_test, gb_pred, average="micro"))
print("Macro F1 Score: ", f1_score_rep(y_test, gb_pred, average="macro"))


knn = KNeighborsClassifier()
knn = knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)


print("****************** kNN prediction results ******************")
print("Accuracy: ", accuracy_score(y_test, knn_pred))
print("Micro F1 Score: ", f1_score_rep(y_test, knn_pred, average="micro"))
print("Macro F1 Score: ", f1_score_rep(y_test, knn_pred, average="macro"))


svm = SVC()
svm = svm.fit(x_train, y_train)
svm_pred = svm.predict(x_test)


print("****************** SVM prediction results ******************")
print("Accuracy: ", accuracy_score(y_test, svm_pred))
print("Micro F1 Score: ", f1_score_rep(y_test, svm_pred, average="micro"))
print("Macro F1 Score: ", f1_score_rep(y_test, svm_pred, average="macro"))

lr = LogisticRegression(random_state=0)
lr = lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)


print("****************** Logistic Regression prediction results ******************")
print("Accuracy: ", accuracy_score(y_test, lr_pred))
print("Micro F1 Score: ", f1_score_rep(y_test, lr_pred, average="micro"))
print("Macro F1 Score: ", f1_score_rep(y_test, lr_pred, average="macro"))

mlp = MLPClassifier()
mlp = mlp.fit(x_train, y_train)
mlp_pred = mlp.predict(x_test)

print("****************** MLP prediction results ******************")
print("Accuracy: ", accuracy_score(y_test, mlp_pred))
print("Micro F1 Score: ", f1_score_rep(y_test, mlp_pred, average="micro"))
print("Macro F1 Score: ", f1_score_rep(y_test, mlp_pred, average="macro"))

rf = RandomForestClassifier()
rf = rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)


print("****************** RF prediction results ******************")
print("Accuracy: ", accuracy_score(y_test, rf_pred))
print("Micro F1 Score: ", f1_score_rep(y_test, rf_pred, average="micro"))
print("Macro F1 Score: ", f1_score_rep(y_test, rf_pred, average="macro"))

ada = AdaBoostClassifier()
ada = ada.fit(x_train, y_train)
ada_pred = ada.predict(x_test)


print("****************** ADA prediction results ******************")
print("Accuracy: ", accuracy_score(y_test, ada_pred))
print("Micro F1 Score: ", f1_score_rep(y_test, ada_pred, average="micro"))
print("Macro F1 Score: ", f1_score_rep(y_test, ada_pred, average="macro"))

gb = GradientBoostingClassifier()
gb = gb.fit(x_train, y_train)
gb_pred = gb.predict(x_test)

print("****************** GB prediction results ******************")
print("Accuracy: ", accuracy_score(y_test, gb_pred))
print("Micro F1 Score: ", f1_score_rep(y_test, gb_pred, average="micro"))
print("Macro F1 Score: ", f1_score_rep(y_test, gb_pred, average="macro"))

xgb = XGBClassifier()
xgb = xgb.fit(x_train, y_train)
xgb_pred = xgb.predict(x_test)

print("****************** XGB prediction results ******************")
print("Accuracy: ", accuracy_score(y_test, xgb_pred))
print("Micro F1 Score: ", f1_score_rep(y_test, xgb_pred, average="micro"))
print("Macro F1 Score: ", f1_score_rep(y_test, xgb_pred, average="macro"))
