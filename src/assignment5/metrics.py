"""
"""
import warnings
from itertools import product
from typing import Dict, Tuple

import pandas as pd
import statsmodels.api
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.simplefilter(action="ignore", category=FutureWarning)


def calculate_predictor_metrics(
    column_list: pd.DataFrame,
    column_list_bins: Dict[str, pd.DataFrame],
    response_series: pd.Series,
) -> pd.DataFrame:
    scores = pd.DataFrame()
    for _, x in column_list.items():
        p_value, t_value = calculate_logarithmic_regression(x, response_series)
        x_bin = column_list_bins[x.name]
        mean_squared_diff_sum = x_bin["mean_squared_diff"].sum()
        mean_squared_diff_sum_w = x_bin["mean_squared_diff_weighted"].sum()
        entry = {
            "name": x.name,
            "p_value": p_value,
            "t_value": t_value,
            "mean_squared_diff_sum": mean_squared_diff_sum,
            "mean_squared_diff_sum_weighted": mean_squared_diff_sum_w,
        }
        scores = scores.append(entry, ignore_index=True)
    scores["rfc_importance"] = calculate_rfc_importance(column_list, response_series)
    scores["score"] = calculate_metric_scores(scores)
    scores = scores.sort_values("score", ascending=False)
    return scores


def calculate_logarithmic_regression(
    x: pd.Series,
    response_series: pd.Series,
) -> Tuple[float, float]:
    df = x.to_frame().join(response_series)
    df = df.dropna()
    x = df[x.name]
    response_series = df[response_series.name]
    p = statsmodels.api.add_constant(x)
    lrm = statsmodels.api.Logit(response_series, p)
    lrm_fit = lrm.fit()
    t_value = round(lrm_fit.tvalues[1], 6)
    p_value = "{:.6e}".format(lrm_fit.pvalues[1])
    return (p_value, t_value)


def calculate_rfc_importance(
    column_list: pd.DataFrame,
    response_series: pd.Series,
) -> pd.Series:
    df = column_list.join(response_series)
    df = df.dropna()
    column_list = df.drop(response_series.name, axis=1)
    response_series = df[response_series.name]
    stdslr = StandardScaler()
    stdslr.fit(column_list)
    column_list = stdslr.transform(column_list)
    rfr = RandomForestClassifier()
    model = rfr.fit(column_list, response_series)
    return model.feature_importances_


def calculate_metric_scores(
    scores: pd.DataFrame,
) -> pd.Series:
    # Scale each metric and average t, weighted difference, and rfc importance
    scaled = pd.DataFrame()
    scaled["t_value"] = scores["t_value"].abs() / scores["t_value"].abs().max()
    scaled["mean_squared_diff_sum_weighted"] = (
        scores["mean_squared_diff_sum_weighted"].abs()
        / scores["mean_squared_diff_sum_weighted"].abs().max()
    )
    scaled["rfc_importance"] = (
        scores["rfc_importance"].abs() / scores["rfc_importance"].abs().max()
    )
    score_series = scaled.mean(axis=1)
    return score_series


def calculate_correlations(
    column_list: pd.DataFrame,
) -> pd.DataFrame:
    corrs = pd.DataFrame()
    column_list = column_list.dropna()
    for c in product(column_list.columns, column_list.columns):
        x0 = column_list[c[0]]
        x1 = column_list[c[1]]
        pearson_coef, _ = pearsonr(x=x0, y=x1)
        corrs = corrs.append(
            other={
                "x0": c[0],
                "x1": c[1],
                "pearson_coef": pearson_coef,
                "abs_coef": abs(pearson_coef),
            },
            ignore_index=True,
        )
    corrs = corrs.sort_values(
        by="abs_coef",
        ascending=False,
    )
    return corrs


def rank_bf(
    column_list_bf: dict,
) -> pd.DataFrame:
    ranks = pd.DataFrame()
    for key, df in column_list_bf.items():
        mean_squared_diff_sum = df["mean_squared_diff"].sum()
        mean_squared_diff_sum_w = df["mean_squared_diff_weighted"].sum()
        entry = {
            "x0": key[0],
            "x1": key[1],
            "mean_squared_diff_sum": mean_squared_diff_sum,
            "mean_squared_diff_sum_weighted": mean_squared_diff_sum_w,
        }
        ranks = ranks.append(entry, ignore_index=True)
    ranks = ranks.sort_values(
        "mean_squared_diff_sum_weighted",
        ascending=False,
    )
    return ranks
