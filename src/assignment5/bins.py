"""
"""
import warnings
from itertools import combinations
from typing import Dict

import pandas as pd

warnings.simplefilter(action="ignore", category=FutureWarning)


def bin_predictor(
    x: pd.Series,
    y: pd.Series,
    n: int = 10,
) -> Dict[str, pd.DataFrame]:
    df = x.to_frame()
    df[y.name] = y
    df["interval"] = pd.cut(x=df[x.name], bins=n)
    bins = pd.DataFrame()
    bins["lower_bin"] = (
        df.groupby("interval")["interval"].first().apply(lambda x: x.left)
    )
    bins["upper_bin"] = (
        df.groupby("interval")["interval"].first().apply(lambda x: x.right)
    )
    bins["bin_center"] = df.groupby("interval")[x.name].median()
    bins["bin_count"] = df.groupby("interval")[y.name].count()
    bins["bin_mean"] = df.groupby("interval")[y.name].mean()
    bins["pop_mean"] = y.mean()
    bins["pop_count"] = y.count()
    bins["mean_squared_diff"] = (bins["bin_mean"] - bins["pop_mean"]) ** 2
    bins["population_proportion"] = bins["bin_count"] / bins["pop_count"]
    bins["mean_squared_diff_weighted"] = (
        bins["population_proportion"] * bins["mean_squared_diff"]
    )
    return bins


if __name__ == "__main__":
    x = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name="x")
    y = pd.Series([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1], name="y")
    print(bin_predictor(x, y, n=5))


def bin_combinations(
    X: pd.DataFrame,
    y: pd.Series,
    n: int = 10,
) -> Dict[str, pd.DataFrame]:
    bf_bins = dict()
    X = X.dropna()
    for c in combinations(X.columns, 2):
        bins = pd.DataFrame()
        df = X[[c[0], c[1]]]
        df[y.name] = y
        df["x0_interval"] = pd.cut(x=df.iloc[:, 0], bins=n)
        df["x1_interval"] = pd.cut(x=df.iloc[:, 1], bins=n)
        bins = pd.DataFrame()
        pop_mean = y.mean()
        pop_count = y.count()
        for x1_ivl in df["x1_interval"].unique():
            for x0_ivl in df["x0_interval"].unique():
                x0mask = df["x0_interval"] == x0_ivl
                x1mask = df["x1_interval"] == x1_ivl
                y_bin = df.loc[(x0mask) & (x1mask)][y.name]
                bin_count = y_bin.count()
                bin_mean = y_bin.mean()
                # bins["pop_mean"] = y.mean()
                # bins["pop_count"] = y.count()
                mean_squared_diff = (bin_mean - pop_mean) ** 2
                population_proportion = bin_count / pop_count
                mean_squared_diff_w = population_proportion * mean_squared_diff
                entry = {
                    "x0_interval": str(x0_ivl),
                    "x1_interval": str(x1_ivl),
                    "bin_count": bin_count,
                    "bin_mean": bin_mean,
                    "pop_mean": pop_mean,
                    "pop_count": pop_count,
                    "mean_squared_diff": mean_squared_diff,
                    "population_proportion": population_proportion,
                    "mean_squared_diff_weighted": mean_squared_diff_w,
                }
                bins = bins.append(
                    entry,
                    ignore_index=True,
                )
        bf_bins[(c[0], c[1])] = bins
    return bf_bins
