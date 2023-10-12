"""
"""
# import statistics
import math
from multiprocessing import cpu_count

import hyperparams
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

rng = np.random.RandomState(0)
KF = KFold(n_splits=2, shuffle=True, random_state=True)
RKF = RepeatedKFold(n_splits=2, n_repeats=2, random_state=2652124)
TSS = TimeSeriesSplit(n_splits=2, max_train_size=None)
n_jobs_cpu = math.floor((cpu_count() / 3))
# n_jobs_cpu=-1
# n_pre_dispatch=math.floor((cpu_count()/3)) + n_jobs_cpu

scoring_list = [
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "f1_micro",
]
"""
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
    result_df = pd.DataFrame()
    for pass_, feature_set in enumerate(feature_sets):
        X = X_
        results = dict()

        for pipeline in PIPELINES:
            name = pipeline.steps[-1][0]

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
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                results[name]["prediction"] += [y_pred]

                results[name]["accuracy_scores"] += [accuracy_score(y_test, y_pred)]
                results[name]["f1_score_micro"] += [
                    f1_score_rep(y_test, y_pred, average="micro")
                ]
                results[name]["f1_score_macro"] += [
                    f1_score_rep(y_test, y_pred, average="macro")
                ]
                results[name]["scores"] += [pipeline.score(X_test, y_test)]

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
                            # scoring=scoring_list,
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
                            # scoring=scoring_list,
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
                            n_splits=6, shuffle=True, random_state=rng
                        )
                        random_search = RandomizedSearchCV(
                            pipeline[name],
                            param_distributions=grid_map,
                            n_iter=30,
                            cv=skf,
                            # pre_dispatch=n_pre_dispatch,
                            n_jobs=n_jobs_cpu,
                            random_state=rng,
                            # scoring=scoring_list,
                            scoring="f1_macro",
                            verbose=5,
                            # error_score='raise',
                        )
                        # print("\nRandomizedSearchCV:\n", random_search.get_params());
                        random_search.fit(X_train, y_train)
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
