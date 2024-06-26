#!/usr/bin/env python
# coding: utf-8

import warnings

import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


df = pd.read_csv("../data/dataset.csv", low_memory=False)


df.head()


df.info()


df.describe()


df.select_dtypes(include=["category", object]).columns


df = df.drop("Avg_user_time", axis=1)


df = df.replace("-", "0")


categorical_columns = ["Protocol", "Service", "anomaly_alert"]
le = LabelEncoder()
for cat_var in categorical_columns:
    df[cat_var] = le.fit_transform(df[cat_var])


df["Scr_ip_bytes"] = df["Scr_ip_bytes"].replace("excel", "0", regex=True)


df = df.replace("?", "0")


df = df.replace("#DIV/0!", "0")


categorical_columns = [
    "Duration",
    "Scr_bytes",
    "Des_bytes",
    "missed_bytes",
    "Scr_pkts",
    "Scr_ip_bytes",
    "Des_pkts",
    "Des_ip_bytes",
    "total_bytes",
    "total_packet",
    "paket_rate",
    "byte_rate",
    "Scr_packts_ratio",
    "Des_pkts_ratio",
    "Scr_bytes_ratio",
    "Des_bytes_ratio",
    "Std_user_time",
    "Avg_nice_time",
    "Std_nice_time",
    "Avg_system_time",
    "Std_system_time",
    "Avg_iowait_time",
    "Std_iowait_time",
    "Avg_ideal_time",
    "Std_ideal_time",
    "Avg_tps",
    "Std_tps",
    "Avg_rtps",
    "Std_rtps",
    "Avg_wtps",
    "Std_wtps",
    "Avg_ldavg_1",
    "Std_ldavg_1",
    "Avg_kbmemused",
    "Std_kbmemused",
    "Avg_num_Proc/s",
    "Std_num_proc/s",
    "Avg_num_cswch/s",
    "std_num_cswch/s",
]


for column in categorical_columns:
    # print('Column name:', column)
    df[column] = df[column].astype(float)


print(df["class1"].value_counts())


drop_columns = ["Date", "Timestamp", "Scr_IP", "Scr_port", "Des_IP", "Des_port"]


df.drop(drop_columns, axis=1, inplace=True)


df.dropna(axis=0, how="any", inplace=True)
df.drop_duplicates(subset=None, keep="first", inplace=True)


# Save the pre-processed dataset
df.to_csv("../data/preprocessed_data.csv", encoding="utf-8", index=False)
