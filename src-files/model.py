import sys
import numpy as np
import pandas as pd
import plotly.express as plotly_express
import plotly.graph_objects as go
import statsmodels.api as sm
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix



def random_forest_importance(response_type, processed_predictor, predictor):
    # Random forest variable importance
    if response_type == "Categorical":
        model = RandomForestClassifier()
    else:
        model = RandomForestRegressor()

    # Replace missing values with mean
    na_cols = processed_predictor.columns[processed_predictor.isna().any()].tolist()
    if na_cols == "":
        pass
    else:
        processed_predictor.loc[:, na_cols] = processed_predictor.loc[
            :, na_cols
        ].fillna(processed_predictor.loc[:, na_cols].mean())

    # Fit random forest model
    model.fit(processed_predictor.iloc[:, 1:], processed_predictor.iloc[:, 0])
    feature_importance = pd.DataFrame(
        model.feature_importances_,
        index=predictor.columns,
        columns=["Random Forest Importance"],
    ).sort_values("Random Forest Importance", ascending=True)

    return feature_importance


def results_table(results, importance):
    # print("\nresults:\n", results)
    # print(importance)
    results.reset_index(inplace=True, drop=True)
    results = pd.concat([results, importance], axis=1)
    # print("\nresults:\n", results)

    with open("./graphs/results.html", "w") as html_open:
        results.to_html(html_open, escape=False)

    return

