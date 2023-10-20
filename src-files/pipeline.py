"""
Import Section
"""
import math
from multiprocessing import cpu_count

import hyperparams
import keras
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
from sklearn import preprocessing
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score as f1_score_rep

# from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    KFold,
    RandomizedSearchCV,
    RepeatedKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, Input, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from xgboost import XGBClassifier

"""
Global Variables section
"""

# Define a random seed here
np.random.seed(1)
PYTHONHASHSEED = 0

# For quicker reqults, determine the number of parallel jobs that can be run
# on different CPUs
n_jobs_cpu = math.floor((cpu_count() / 3))
if n_jobs_cpu < 1:
    n_jobs_cpu = None
n_jobs_cpu = None

# Number of iterations for Neural networks
n_epochs = 2  # 30
n_cv = 3

# Random state
rng = np.random.RandomState(0)

# Fold algirithms
KF = KFold(n_splits=2, shuffle=True, random_state=True)
RKF = RepeatedKFold(n_splits=2, n_repeats=2, random_state=2652124)
TSS = TimeSeriesSplit(n_splits=2, max_train_size=None)

# define callbacks
# early_stop = EarlyStopping(monitor = 'val_accuracy', mode = 'max', patience=5, restore_best_weights=True)
early_stop = EarlyStopping(
    monitor="val_loss", mode="max", patience=5, restore_best_weights=True
)


"""
Generic Keras Model
This is a model generating function so that we can search over neural net
parameters and architecture
"""


def create_keras_model(
    nl1=1,
    nl2=1,
    nl3=1,
    nn1=1000,
    nn2=500,
    nn3=200,
    learning_rate=0.01,
    decay=0.0,
    l1=0.01,
    l2=0.01,
    act="relu",
    dropout=0,
    input_shape=61,
    output_shape=2,
):
    self_optimizer = Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="Adam",
    )
    reg = keras.regularizers.l1_l2(l1=l1, l2=l2)

    model = Sequential()

    # for the firt layer we need to specify the input dimensions
    first = True

    for i in range(nl1):
        if first:
            model.add(
                Dense(
                    nn1, input_dim=input_shape, activation=act, kernel_regularizer=reg
                )
            )
            first = False
        else:
            model.add(Dense(nn1, activation=act, kernel_regularizer=reg))
        if dropout != 0:
            model.add(Dropout(dropout))

    for i in range(nl2):
        if first:
            model.add(
                Dense(
                    nn2, input_dim=input_shape, activation=act, kernel_regularizer=reg
                )
            )
            first = False
        else:
            model.add(Dense(nn2, activation=act, kernel_regularizer=reg))
        if dropout != 0:
            model.add(Dropout(dropout))

    for i in range(nl3):
        if first:
            model.add(
                Dense(
                    nn3, input_dim=input_shape, activation=act, kernel_regularizer=reg
                )
            )
            first = False
        else:
            model.add(Dense(nn3, activation=act, kernel_regularizer=reg))
        if dropout != 0:
            model.add(Dropout(dropout))

    model.add(Dense(output_shape, activation="sigmoid"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=self_optimizer,
        metrics=["accuracy"],
    )
    return model


"""
CNN Model
"""


def create_cnn_model(input_shape=[61, 1], output_shape=2, num_classes=2):
    # self_optimizer = SGD(learning_rate=0.01) ### divide by 10 if learning stops after some epochs
    self_optimizer = Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="Adam",
    )
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
    cnn_model.compile(
        loss="categorical_crossentropy", optimizer=self_optimizer, metrics=["accuracy"]
    )
    # cnn_model.compile(loss = "sparse_categorical_crossentropy", optimizer=self_optimizer, metrics=['accuracy'])
    return cnn_model


"""
DNN Model
"""


def create_dnn_model(
    input_shape=[
        61,
    ],
    output_shape=2,
    num_classes=2,
):
    # self_optimizer = SGD(learning_rate=0.01)
    self_optimizer = Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="Adam",
    )
    dnn_model = Sequential()
    dnn_model.add(Input(shape=input_shape))
    dnn_model.add(Dense(units=30, activation="relu"))
    dnn_model.add(Dense(units=20, activation="relu"))
    dnn_model.add(Dense(units=num_classes, activation="softmax"))
    # dnn_model.compile(loss = "sparse_categorical_crossentropy", optimizer=self_optimizer, metrics=['accuracy'])
    dnn_model.compile(
        loss="categorical_crossentropy", optimizer=self_optimizer, metrics=["accuracy"]
    )
    return dnn_model


"""
Pipelines
"""
PIPELINES = [
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("RandomForestClassifier", RandomForestClassifier()),
        ],
    ),
    Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "LogisticRegression",
                LogisticRegression(),
            ),
        ],
    ),
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("LinearSVC", LinearSVC()),
        ],
    ),
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("GaussianNB", GaussianNB()),
        ],
    ),
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("XGBClassifier", XGBClassifier()),
        ],
    ),
    Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "GradientBoostingClassifier",
                GradientBoostingClassifier(),
            ),
        ],
    ),
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("AdaBoostClassifier", AdaBoostClassifier()),
        ],
    ),
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("DecisionTreeClassifier", DecisionTreeClassifier()),
        ],
    ),
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("KNeighborsClassifier", KNeighborsClassifier()),
        ],
    ),
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("SVC", SVC()),
        ],
    ),
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("MLPClassifier", MLPClassifier()),
        ],
    ),
    Pipeline(
        [
            (
                "CNNClassifier",
                KerasClassifier(
                    model=create_cnn_model, epochs=6, batch_size=64, verbose=1
                ),
            ),
        ],
    ),
    Pipeline(
        [
            (
                "DNNClassifier",
                KerasClassifier(
                    model=create_dnn_model, epochs=6, batch_size=64, verbose=1
                ),
            ),
        ],
    ),
    Pipeline(
        [
            (
                "KerasClassifier",
                KerasClassifier(
                    model=create_keras_model, epochs=6, batch_size=64, verbose=1
                ),
            ),
        ],
    ),
]


def try_models(
    X_: pd.DataFrame,
    y: pd.Series,
    feature_sets,
    path,
    title,
    split_type,
    hyper_tuning_en,
    op_type,
) -> None:
    # result_df = pd.DataFrame()
    for pass_, feature_set in enumerate(feature_sets):
        X = X_
        results = dict()

        for pipeline in PIPELINES:
            name = pipeline.steps[-1][0]
            if (
                name == "KerasClassifier"
                or name == "CNNClassifier"
                or name == "DNNClassifier"
            ):
                isNeural = 1
            else:
                isNeural = 0

            print(name, " is using:\n", pipeline[name].get_params())
            results[name] = dict()
            results[name]["scores"] = list()
            results[name]["accuracy_scores"] = list()
            results[name]["f1_score_micro"] = list()
            results[name]["f1_score_macro"] = list()
            results[name]["prediction"] = list()
            results[name]["probs"] = list()
            results[name]["confusion"] = list()

            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
            if split_type == "tss":
                # print("Time series splitting")
                enumeration_data = TSS.split(X)
            elif split_type == "kfold":
                # print("K-Fold splitting")
                enumeration_data = KF.split(X)
            else:
                # print("Repeated K-Fold splitting")
                enumeration_data = RKF.split(X)

            for _, split in enumerate(enumeration_data):
                train_index, test_index = split
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                if isNeural:
                    scaler = preprocessing.MinMaxScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    if name == "KerasClassifier":
                        pipeline.fit(
                            X_train,
                            to_categorical(y_train),
                            KerasClassifier__epochs=n_epochs,
                            KerasClassifier__validation_data=(
                                X_test,
                                to_categorical(y_test),
                            ),
                            KerasClassifier__batch_size=64,
                            KerasClassifier__validation_split=0.2,
                            KerasClassifier__callbacks=[early_stop],
                        )
                        results[name]["scores"] += [
                            pipeline.score(X_test, to_categorical(y_test))
                        ]
                        y_pred = pipeline.predict(X_test)
                        results[name]["scores"] += [pipeline.score(X_test, y_test)]
                    elif name == "CNNClassifier":
                        x_train = np.copy(X_train_scaled)
                        x_test = np.copy(X_test_scaled)
                        label_encoder = LabelEncoder()
                        y_train = label_encoder.fit_transform(y_train)
                        y_test = label_encoder.transform(y_test)
                        # Reshape training and test dataa for DL model training
                        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
                        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
                        pipeline.fit(
                            x_train,
                            to_categorical(y_train),
                            CNNClassifier__epochs=n_epochs,
                            CNNClassifier__validation_data=(
                                x_test,
                                to_categorical(y_test),
                            ),
                            CNNClassifier__batch_size=64,
                            CNNClassifier__validation_split=0.2,
                            CNNClassifier__callbacks=[early_stop],
                        )

                        y_hat = pipeline.predict(x_test)
                        y_hat = np.argmax(y_hat, axis=-1)
                        y_pred = y_hat
                        results[name]["scores"] += [
                            pipeline.score(x_test, to_categorical(y_test))
                        ]
                    elif name == "DNNClassifier":
                        x_train = np.copy(X_train_scaled)
                        x_test = np.copy(X_test_scaled)
                        label_encoder = LabelEncoder()
                        y_train = label_encoder.fit_transform(y_train)
                        y_test = label_encoder.transform(y_test)
                        # Reshape training and test dataa for DL model training
                        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
                        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
                        pipeline.fit(
                            x_train,
                            to_categorical(y_train),
                            DNNClassifier__epochs=n_epochs,
                            DNNClassifier__validation_data=(
                                x_test,
                                to_categorical(y_test),
                            ),
                            DNNClassifier__batch_size=64,
                            DNNClassifier__validation_split=0.2,
                            DNNClassifier__callbacks=[early_stop],
                        )
                        y_hat = pipeline.predict(x_test)
                        y_hat = np.argmax(y_hat, axis=-1)
                        y_pred = y_hat
                        results[name]["scores"] += [
                            pipeline.score(x_test, to_categorical(y_test))
                        ]
                else:
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    results[name]["scores"] += [pipeline.score(X_test, y_test)]

                results[name]["prediction"] += [y_pred]
                results[name]["accuracy_scores"] += [accuracy_score(y_test, y_pred)]
                results[name]["f1_score_micro"] += [
                    f1_score_rep(y_test, y_pred, average="micro")
                ]
                results[name]["f1_score_macro"] += [
                    f1_score_rep(y_test, y_pred, average="macro")
                ]

                print(name, "classification: ", classification_report(y_pred, y_test))

                if hyper_tuning_en == "half-gradient":
                    grid_map = hyperparams.get_params_grid(name, hyper_tuning_en)
                    if len(grid_map):
                        halfgridsearch = HalvingGridSearchCV(
                            pipeline[name],
                            param_grid=grid_map,
                            resource="n_samples",
                            cv=4,
                            max_resources=15,
                            random_state=0,
                            n_jobs=n_jobs_cpu,
                            scoring="f1_macro",
                            # error_score='raise',
                        )
                        # print("\nHalfGridSearchCV:\n",halfgridsearch.get_params());
                        halfgridsearch.fit(X_train, y_train)
                        print(
                            name,
                            ": ",
                            "The best estimator across ALL searched params:",
                            halfgridsearch.best_estimator_,
                        )
                        print(
                            name,
                            ": ",
                            "The best score across ALL searched params:",
                            halfgridsearch.best_score_,
                        )
                        print(
                            name,
                            ": ",
                            "The best parameters across ALL searched params:",
                            halfgridsearch.best_params_,
                        )
                        # print("\n", halfgridsearch.cv_results_)
                    else:
                        print("list is empty for : ", name)

                elif hyper_tuning_en == "gradient":
                    grid_map = hyperparams.get_params_grid(name, hyper_tuning_en)
                    if len(grid_map):
                        grid_search = GridSearchCV(
                            pipeline[name],
                            param_grid=grid_map,
                            cv=5,
                            n_jobs=n_jobs_cpu,
                            scoring="f1_macro",
                            # error_score='raise',
                        )
                        # print("\nGridSearchCV:\n",grid_search.get_params());
                        grid_search.fit(X_train, y_train)
                        print(
                            name,
                            ": ",
                            "The best estimator across ALL searched params:",
                            grid_search.best_estimator_,
                        )
                        print(
                            name,
                            ": ",
                            "The best score across ALL searched params:",
                            grid_search.best_score_,
                        )
                        print(
                            name,
                            ": ",
                            "The best parameters across ALL searched params:",
                            grid_search.best_params_,
                        )
                        # print("\n",grid_search.cv_results_)
                    else:
                        print("list is empty for : ", name)

                elif hyper_tuning_en == "random":
                    grid_map = hyperparams.get_params_grid(name, hyper_tuning_en)
                    if len(grid_map):
                        skf = StratifiedKFold(
                            n_splits=2, shuffle=True, random_state=rng
                        )

                        if isNeural == 1:
                            random_search = RandomizedSearchCV(
                                pipeline[name],
                                param_distributions=grid_map,
                                n_iter=2,
                                cv=n_cv,
                                n_jobs=n_jobs_cpu,
                                random_state=rng,
                                scoring="f1_macro",
                                verbose=1,
                                # error_score='raise',
                            )
                            if name == "KerasClassifier":
                                random_search.fit(
                                    X_train,
                                    to_categorical(y_train),
                                )
                                random_search.fit(X_train, y_train)
                                if hasattr(pipeline, "predict_proba"):
                                    results[name]["probs"] += [
                                        pipeline.predict_proba(
                                            X_test,
                                        )[:, 1]
                                    ]
                                else:
                                    results[name]["model_probs"] = None
                                    results[name]["confusion"] += [
                                        confusion_matrix(y_test, y_pred),
                                    ]
                            elif name == "CNNClassifier":
                                x_train = np.copy(X_train_scaled)
                                x_test = np.copy(X_test_scaled)
                                label_encoder = LabelEncoder()
                                y_train = label_encoder.fit_transform(y_train)
                                y_test = label_encoder.transform(y_test)
                                # Reshape training and test dataa for DL model training
                                x_train = x_train.reshape(
                                    x_train.shape[0], x_train.shape[1], 1
                                )
                                x_test = x_test.reshape(
                                    x_test.shape[0], x_test.shape[1], 1
                                )
                                random_search.fit(
                                    x_train,
                                    to_categorical(y_train),
                                )

                                y_hat = pipeline.predict(x_test)
                                y_hat = np.argmax(y_hat, axis=-1)
                                y_pred = y_hat
                                random_search.fit(x_train, to_categorical(y_train))
                                if hasattr(pipeline, "predict_proba"):
                                    results[name]["probs"] += [
                                        pipeline.predict_proba(
                                            x_test,
                                        )[:, 1]
                                    ]
                                else:
                                    results[name]["model_probs"] = None
                                    results[name]["confusion"] += [
                                        confusion_matrix(y_test, y_pred),
                                    ]
                            elif name == "DNNClassifier":
                                x_train = np.copy(X_train_scaled)
                                x_test = np.copy(X_test_scaled)
                                label_encoder = LabelEncoder()
                                y_train = label_encoder.fit_transform(y_train)
                                y_test = label_encoder.transform(y_test)
                                # Reshape training and test dataa for DL model training
                                x_train = x_train.reshape(
                                    x_train.shape[0], x_train.shape[1], 1
                                )
                                x_test = x_test.reshape(
                                    x_test.shape[0], x_test.shape[1], 1
                                )
                                random_search.fit(
                                    x_train,
                                    to_categorical(y_train),
                                )
                                y_hat = pipeline.predict(x_test)
                                y_hat = np.argmax(y_hat, axis=-1)
                                y_pred = y_hat
                                random_search.fit(x_train, to_categorical(y_train))
                                if hasattr(pipeline, "predict_proba"):
                                    results[name]["probs"] += [
                                        pipeline.predict_proba(
                                            x_test,
                                        )[:, 1]
                                    ]
                                else:
                                    results[name]["model_probs"] = None
                                    results[name]["confusion"] += [
                                        confusion_matrix(y_test, y_pred),
                                    ]
                        else:
                            random_search = RandomizedSearchCV(
                                pipeline[name],
                                param_distributions=grid_map,
                                n_iter=40,
                                cv=skf,
                                n_jobs=n_jobs_cpu,
                                random_state=rng,
                                scoring="f1_macro",
                                verbose=5,
                                # error_score='raise',
                            )
                            # print("\nRandomizedSearchCV:\n", random_search.get_params());
                            random_search.fit(X_train, y_train)
                            if hasattr(pipeline, "predict_proba"):
                                results[name]["probs"] += [
                                    pipeline.predict_proba(
                                        X_test,
                                    )[:, 1]
                                ]
                            else:
                                results[name]["model_probs"] = None
                                results[name]["confusion"] += [
                                    confusion_matrix(y_test, y_pred),
                                ]
                        print(
                            name,
                            ": ",
                            "The best estimator across ALL searched params:",
                            random_search.best_estimator_,
                        )
                        print(
                            name,
                            ": ",
                            "The best score across ALL searched params:",
                            random_search.best_score_,
                        )
                        print(
                            name,
                            ": ",
                            "The best parameters across ALL searched params:",
                            random_search.best_params_,
                        )
                        # print("\n",random_search.cv_results_['params'])
                    else:
                        print("list is empty for : ", name)

            print("############ Results Start ###############")
            print(name)
            print("scores:", results[name]["scores"])
            print("Accuracy: ", results[name]["accuracy_scores"])
            print("Micro F1 Score: ", results[name]["f1_score_micro"])
            print("Macro F1 Score: ", results[name]["f1_score_macro"])
            print("########### Results End ################")
            # final_results = pd.DataFrame(results[name])
            # filename =  op_type + name + ".csv"
            # final_results.to_csv( filename, index=False)
        """
        for model, model_dict in results.items():
            for run, score in enumerate(model_dict["scores"]):
                row = {
                    "model": model,
                    "feature_set": feature_set,
                    "score": score,
                }
                result_df = result_df.append(row, ignore_index=True)
                # Plot confusion matrix for each run
                # https://stackoverflow.com/questions/60860121
                labels = [0, 1]
                cm = model_dict["confusion"][run]
                data = go.Heatmap(z=cm, y=labels, x=labels, colorscale="BrBg")
                annotations = []
                for i, row in enumerate(cm):
                    for j, value in enumerate(row):
                        annotations.append(
                            {
                                "x": labels[i],
                                "y": labels[j],
                                "font": {"color": "black"},
                                "text": str(value),
                                "xref": "x1",
                                "yref": "y1",
                                "showarrow": False,
                            }
                        )
                layout = {
                    "title": f"CF for {model}, run {i}",
                    "xaxis": {"title": "Predicted value"},
                    "yaxis": {"title": "Real value"},
                    "annotations": annotations,
                }
                fig = go.Figure(
                    data=data,
                    layout=layout,
                )
                fig.write_html(f"{path}/{pass_}_{model}_cm_{run}_{op_type}.html")
        """
    """
    fig = px.box(
        result_df,
        x="feature_set",
        y="score",
        color="model",
        title=f"Model scores from {title} features",
    )

    fig.add_shape(
        type="line",
        x0=0,
        y0=y,
        x1=1,
        y1=y,
        line={"color": "black"},
        xref="paper",
        yref="y",
    )

    fig.write_html(f"{path}/score_models_{op_type}.html")
    """
    """
    fig = px.box(
        result_df,
        x="feature_set",
        y="accuracy_scores",
        color="model",
        title=f"Model scores from {title} features",
    )

    fig.add_shape(
        type="line",
        x0=0,
        y0=y,
        x1=1,
        y1=y,
        line={"color": "black"},
        xref="paper",
        yref="y",
    )

    fig.write_html(f"{path}/accuracy_models_{op_type}.html")

    fig = px.box(
        result_df,
        x="feature_set",
        y="f1_score_micro",
        color="model",
        title=f"Model scores from {title} features",
    )

    fig.add_shape(
        type="line",
        x0=0,
        y0=y,
        x1=1,
        y1=y,
        line={"color": "black"},
        xref="paper",
        yref="y",
    )
    fig.write_html(f"{path}/f1_micro_models_{op_type}.html")
    fig = px.box(
        result_df,
        x="feature_set",
        y="f1_score_macro",
        color="model",
        title=f"Model scores from {title} features",
    )

    fig.add_shape(
        type="line",
        x0=0,
        y0=y,
        x1=1,
        y1=y,
        line={"color": "black"},
        xref="paper",
        yref="y",
    )
    fig.write_html(f"{path}/f1_macro_models_{op_type}.html")
    """
    return


if __name__ == "__main__":
    pass
