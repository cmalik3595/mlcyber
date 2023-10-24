#!/usr/bin/env python
# coding: utf-8
import warnings

import numpy as np
import pandas as pd

# Tensorflow and Keras
import tensorflow as tf
from keras.layers import Dense, Flatten
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score as f1_score_rep
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Conv1D, Input, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

warnings.filterwarnings("ignore")
df = pd.read_csv("../data/preprocessed_data.csv", low_memory=False)


y = df["class1"]  # get the most detailed class labels (class1 > class2 > class3)
X = df.drop(columns=["class1", "class2", "class3"])


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
# Reshape training and test dataa for DL model training
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
input_shape = x_train.shape[1:]
num_classes = len(np.unique(y_train))


#  DL Models

# Setting seed for reproducability
np.random.seed(1)
PYTHONHASHSEED = 0


#  Convolutional Neural Network


cnn_model = Sequential()
cnn_model.add(
    Conv1D(
        filters=20,
        kernel_size=4,
        strides=2,
        padding="valid",
        activation="relu",
        input_shape=input_shape,
    )
)
cnn_model.add(MaxPooling1D())
cnn_model.add(
    Conv1D(filters=20, kernel_size=4, strides=2, padding="same", activation="relu")
)
cnn_model.add(
    Conv1D(filters=3, kernel_size=2, strides=1, padding="same", activation="relu")
)
cnn_model.add(Flatten())
cnn_model.add(Dense(units=100, activation="relu"))
cnn_model.add(Dense(units=num_classes, activation="softmax"))

opt = SGD(lr=0.01)  # divide by 10 if learning stops after some epochs
cnn_model.compile(
    loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
)


hist = cnn_model.fit(
    x_train,
    y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)
    ],
)


print(x_test.shape)
y_hat = cnn_model.predict(x_test)
y_hat = np.argmax(y_hat, axis=-1)


print("****************** 1-D CNN prediction results ******************")
print("Accuracy: ", accuracy_score(y_test, y_hat))
print("Micro F1 Score: ", f1_score_rep(y_test, y_hat, average="micro"))
print("Macro F1 Score: ", f1_score_rep(y_test, y_hat, average="macro"))


#  Deep Neural Network


x_train = np.copy(X_train_scaled)
x_test = np.copy(X_test_scaled)
input_shape = x_train.shape[1:]
print(input_shape)


dnn_model = Sequential()
dnn_model.add(Input(shape=input_shape))
dnn_model.add(Dense(units=30, activation="relu"))
dnn_model.add(Dense(units=20, activation="relu"))
dnn_model.add(Dense(units=num_classes, activation="softmax"))

opt = SGD(lr=0.01)
dnn_model.compile(
    loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
)


hist = dnn_model.fit(
    x_train,
    y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)
    ],
)


y_hat = dnn_model.predict(x_test)
y_hat = np.argmax(y_hat, axis=-1)


print("****************** DNN prediction results ******************")
print("Accuracy: ", accuracy_score(y_test, y_hat))
print("Micro F1 Score: ", f1_score_rep(y_test, y_hat, average="micro"))
print("Macro F1 Score: ", f1_score_rep(y_test, y_hat, average="macro"))


#  Saving and loading DNN models


# Save the CNN model
# cnn_model.save('CNN-X-IIoT.h5')


# Load the CNN model
# dcnn_model = tensorflow.keras.models.load_model('CNN-X-IIoT.h5')
