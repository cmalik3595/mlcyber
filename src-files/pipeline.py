"""
"""
# import statistics

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


import xgboost

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsClassifier


from sklearn.metrics import f1_score as f1_score_rep
from sklearn.metrics import accuracy_score


TSS = TimeSeriesSplit(n_splits=10, max_train_size=None)

PIPELINES = [
    Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "LogisticRegression",
                LogisticRegression(
                    multi_class="multinomial",
                    max_iter = 1500
                ),
            ),
        ],
    ),
    Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "LinearSVC", LinearSVC(
                    max_iter=100000,
                    dual='auto'
                    )
             ),
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
            ("RandomForestClassifier", RandomForestClassifier()),
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
            ("GradientBoostingClassifier", GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=1.0, 
                max_depth=1, 
                random_state=0
                )
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
            ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=0)),
        ],
    ),
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("KNeighborsClassifier", KNeighborsClassifier(n_neighbors=3)),
        ],
    ),
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("SVC", SVC(kernel='rbf', C = 1)),
        ],
    ),
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("SVC", SVC(kernel='rbf', C = 1)),
        ],
    ),
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("MLPClassifier", MLPClassifier(random_state=1, max_iter=100)),
        ],
    ),
]


def try_models(
    X_: pd.DataFrame,
    y: pd.Series,
    feature_sets,
    path,
    title,
) -> None:
    result_df = pd.DataFrame()
    for pass_, feature_set in enumerate(feature_sets):
        X = X_
        results = dict()

        for pipeline in PIPELINES:
            name = pipeline.steps[-1][0]
            results[name] = dict()
            results[name]["scores"] = list()
            results[name]["accuracy_scores"] = list()
            results[name]["f1_score_micro"] = list()
            results[name]["f1_score_macro"] = list()
            results[name]["prediction"] = list()
            results[name]["probs"] = list()
            results[name]["confusion"] = list()

            for _, split in enumerate(TSS.split(X)):
                train_index, test_index = split
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                results[name]["prediction"] += [y_pred]

                results[name]["accuracy_scores"] += [accuracy_score(y_test, y_pred)]
                results[name]["f1_score_micro"] += [f1_score_rep(y_test, y_pred, average="micro")]
                results[name]["f1_score_macro"] += [f1_score_rep(y_test, y_pred, average="macro")]
                results[name]["scores"] += [pipeline.score(X_test, y_test)]

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

        print("###########################")
        print(name)
        print("scores:", results[name]["scores"])
        print("Accuracy: ", results[name]["accuracy_scores"])
        print("Micro F1 Score: ",results[name]["f1_score_micro"])
        print("Macro F1 Score: ", results[name]["f1_score_macro"])
        print("###########################")

        for model, model_dict in results.items():
            for run, score in enumerate(model_dict["scores"]):
                row = {
                    "model": model,
                    "feature_set": ", ".join(feature_set),
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
                fig.write_html(f"{path}/plots/{pass_}_{model}_cm_{run}.html")

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
        y0=y["class3"],
        x1=1,
        y1=y["class3"],
        line={"color": "black"},
        xref="paper",
        yref="y",
    )
    fig.write_html(f"{path}/models.html")
    return


if __name__ == "__main__":
    pass
