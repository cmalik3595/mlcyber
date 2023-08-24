#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import warnings
from time import time

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score as f1_score_rep

# In[2]:


df = pd.read_csv("preprocessed_data.csv", low_memory=False)


# In[ ]:


y = df["class1"]  # get the most detailed class labels (class1 > class2 > class3)
X = df.drop(columns=["class1", "class2", "class3"])
del df


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)
del X, y


# In[ ]:


scaler = preprocessing.MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[ ]:


x_train = np.copy(X_train_scaled)
x_test = np.copy(X_test_scaled)


# In[ ]:


label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)


# In[ ]:


### Decision Tree
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(x_train, y_train)
dt_pred = clf.predict(x_test)


# In[ ]:


print("****************** Decision Tree prediction results ******************")
print("Accuracy: ", accuracy_score(y_test, dt_pred))
print("Micro F1 Score: ", f1_score_rep(y_test, dt_pred, average="micro"))
print("Macro F1 Score: ", f1_score_rep(y_test, dt_pred, average="macro"))


# In[ ]:


### Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

gb = GaussianNB()
gb = gb.fit(x_train, y_train)
gb_pred = gb.predict(x_test)


# In[ ]:


print("****************** Gaussian NB prediction results ******************")
print("Accuracy: ", accuracy_score(y_test, gb_pred))
print("Micro F1 Score: ", f1_score_rep(y_test, gb_pred, average="micro"))
print("Macro F1 Score: ", f1_score_rep(y_test, gb_pred, average="macro"))


# In[ ]:


### k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn = knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)


# In[ ]:


print("****************** kNN prediction results ******************")
print("Accuracy: ", accuracy_score(y_test, knn_pred))
print("Micro F1 Score: ", f1_score_rep(y_test, knn_pred, average="micro"))
print("Macro F1 Score: ", f1_score_rep(y_test, knn_pred, average="macro"))


# In[ ]:


### Support Vector Machine
from sklearn.svm import SVC

svm = SVC(kernel="rbf", C=1)
svm = svm.fit(x_train, y_train)
svm_pred = svm.predict(x_test)


# In[ ]:


print("****************** SVM prediction results ******************")
print("Accuracy: ", accuracy_score(y_test, svm_pred))
print("Micro F1 Score: ", f1_score_rep(y_test, svm_pred, average="micro"))
print("Macro F1 Score: ", f1_score_rep(y_test, svm_pred, average="macro"))


# In[ ]:


### Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=0)
lr = lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)


# In[ ]:


print("****************** Logistic Regression prediction results ******************")
print("Accuracy: ", accuracy_score(y_test, lr_pred))
print("Micro F1 Score: ", f1_score_rep(y_test, lr_pred, average="micro"))
print("Macro F1 Score: ", f1_score_rep(y_test, lr_pred, average="macro"))


# In[ ]:


### Multi-layer Perceptron
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(random_state=1, max_iter=100)
mlp = mlp.fit(x_train, y_train)
mlp_pred = mlp.predict(x_test)


# In[ ]:


print("****************** MLP prediction results ******************")
print("Accuracy: ", accuracy_score(y_test, mlp_pred))
print("Micro F1 Score: ", f1_score_rep(y_test, mlp_pred, average="micro"))
print("Macro F1 Score: ", f1_score_rep(y_test, mlp_pred, average="macro"))


# In[ ]:


### Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth=5, random_state=0)
rf = rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)


# In[ ]:


print("****************** RF prediction results ******************")
print("Accuracy: ", accuracy_score(y_test, rf_pred))
print("Micro F1 Score: ", f1_score_rep(y_test, rf_pred, average="micro"))
print("Macro F1 Score: ", f1_score_rep(y_test, rf_pred, average="macro"))


# In[ ]:


### AdaBoost Classifier
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(n_estimators=100)
ada = ada.fit(x_train, y_train)
ada_pred = ada.predict(x_test)


# In[ ]:


print("****************** ADA prediction results ******************")
print("Accuracy: ", accuracy_score(y_test, ada_pred))
print("Micro F1 Score: ", f1_score_rep(y_test, ada_pred, average="micro"))
print("Macro F1 Score: ", f1_score_rep(y_test, ada_pred, average="macro"))


# In[ ]:


### Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0
)
gb = gb.fit(x_train, y_train)
gb_pred = gb.predict(x_test)


# In[ ]:


print("****************** GB prediction results ******************")
print("Accuracy: ", accuracy_score(y_test, gb_pred))
print("Micro F1 Score: ", f1_score_rep(y_test, gb_pred, average="micro"))
print("Macro F1 Score: ", f1_score_rep(y_test, gb_pred, average="macro"))


# In[ ]:


### XGBoost
from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb = xgb.fit(x_train, y_train)
xgb_pred = xgb.predict(x_test)


# In[ ]:


print("****************** XGB prediction results ******************")
print("Accuracy: ", accuracy_score(y_test, xgb_pred))
print("Micro F1 Score: ", f1_score_rep(y_test, xgb_pred, average="micro"))
print("Macro F1 Score: ", f1_score_rep(y_test, xgb_pred, average="macro"))
