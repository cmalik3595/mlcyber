import os
import os.path

import pandas as pd


def pre_processing(data_frames: pd.DataFrame) -> pd.DataFrame:
    data_frames.select_dtypes(include=["category", object]).columns
    data_frames = data_frames.replace("-", "0")
    data_frames = data_frames.replace("?", "0")
    data_frames = data_frames.replace("#DIV/0!", "0")
    data_frames = data_frames.replace("FALSE", "0")
    data_frames = data_frames.replace("TRUE", "1")
    data_frames.columns = data_frames.columns.str.replace("/", "_")

    data_frames = data_frames.drop("class1", axis=1)
    data_frames = data_frames.drop("class2", axis=1)
    # Drop the columns which contains only a single value. They bear no impact
    # on the output
    for col in data_frames.columns:
        if len(data_frames[col].unique()) == 1:
            data_frames.drop(col, inplace=True, axis=1)

    data_frames = data_frames.dropna(axis=1, how="any")
    data_frames = data_frames.drop_duplicates(subset=None, keep="first")
    data_frames["Scr_ip_bytes"] = data_frames["Scr_ip_bytes"].replace(
        "excel", "0", regex=True
    )

    return data_frames


def load_file():
    file_path = "../data/dataset.csv"
    while not os.path.exists(file_path):
        file_path = input("\nEnter complete file path for the input:\n")
    # Make a folder to store the plots
    if not os.path.exists("plots"):
        os.makedirs("plots")

    if ".xlsx" in file_path:
        data_frames = pd.read_excel("Test.xlsx")
    else:
        data_frames = pd.read_csv(file_path, low_memory=False)

    data_frames = pre_processing(data_frames)

    data_frames.to_csv("../data/preprocceded.csv", encoding="utf-8", index=False)

    data_frames = pd.read_csv("../data/preprocceded.csv", low_memory=False)

    # This will constitute the feature list
    column_names = data_frames.columns.values.tolist()

    # Default response feature
    response_feature = "class3"

    # Search for the response feature in the column list. Ask the user for the intended column
    # to be used as a response feature.
    while response_feature not in column_names and len(response_feature) != 1:
        response_feature = input("\nEnter one response feature variable name:\n")
    else:
        pass

    prediction_variables = []

    prediction_variables = data_frames.drop(response_feature, axis=1)

    return data_frames, response_feature, prediction_variables
