import os
import os.path
import sys

import numpy as np
import pandas as pd
import plotly.express as plotly_express
import plotly.graph_objects as go
import statsmodels.api as sm
#import pdb
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix

#pdb.set_trace()
#pdb.continue()

# Decision rules for categorical:
# - If string
# - If unique values make up less than 5% of total obs
def continuous_or_categorical_result(data_frames, response_columns):
    resp_string_check = isinstance(response_columns.values, str)
    resp_unique_ratio = len(np.unique(response_columns.values)) / len(
        response_columns.values
    )

    if resp_string_check or resp_unique_ratio < 0.05:
        return "Categorical"
    else:
        return "Continuous"


def continuous_or_categorical_predictor(predictor_data):
    predictor_string_check = isinstance(predictor_data, str)
    pred_unique_ratio = len(predictor_data.unique()) / len(predictor_data)
    if predictor_string_check or pred_unique_ratio < 0.05:
        return "Categorical"
    else:
        return "Continuous"


def pre_processing(data_frames):
    data_frames.select_dtypes(include=["category", object]).columns
    data_frames = data_frames.drop("class1", axis=1)
    data_frames = data_frames.drop("class2", axis=1)
    data_frames = data_frames.replace("-", "0")
    data_frames = data_frames.replace("?", "0")
    data_frames = data_frames.replace("#DIV/0!", "0")
    data_frames["Scr_ip_bytes"] = data_frames["Scr_ip_bytes"].replace(
        "excel", "0", regex=True
    )

    return data_frames


def load_file():
    file_path = ""
    # while not os.path.exists(file_path):
    #    file_path = input("\nEnter complete file path for the input:\n")
    file_path = "../data/dataset.csv"
    # Make a folder to store the plots
    if not os.path.exists("graphs"):
        os.makedirs("graphs")

    if ".xlsx" in file_path:
        data_frames = pd.read_excel("Test.xlsx")
    else:
        data_frames = pd.read_csv(file_path, low_memory=False)

    #data_frames.head()
    #data_frames.info()

    data_frames = pre_processing(data_frames)

    # This will constitute the feature list
    column_names = data_frames.columns.values.tolist()
    # print(column_names)

    # Default response feature
    response_feature = "class3"

    # Search for the response feature in the column list. Ask the user for the intended column
    # to be used as a response feature.
    while response_feature not in column_names and len(response_feature) != 1:
        response_feature = input("\nEnter one response feature variable name:\n")
    else:
        pass

    prediction_variables = []
    data_frames = data_frames.dropna(axis=1, how="any")
    prediction_variables = data_frames.drop(response_feature, axis=1)

    return data_frames, response_feature, prediction_variables


def process_response(data_frames, response):
    # Check var type of response
    # print("entering process_response")
    response_columns = data_frames[response]
    # print(response_columns)
    response_var_type = continuous_or_categorical_result(data_frames, response_columns)
    # print("Resp Var")
    # print(response_var_type)
    if response_var_type == "Categorical":
        response_type = "Categorical"

        # Plot histogram
        resp_plot = plotly_express.histogram(response_columns)
        file_name = "graphs/categorical_response.html"
        resp_plot.write_html(file=file_name, include_plotlyjs="cdn")

        # print("Before processing")
        # print(response_columns)
        # Encode
        response_columns = pd.Categorical(
            response_columns, categories=response_columns.unique()
        )
        response_columns, resp_labels = pd.factorize(response_columns)

        response_columns = pd.DataFrame(response_columns, columns=[response])
        response_columns_uncoded = data_frames[response]

        # print("After processing")
        # print(response_columns)

    else:
        response_type = "Continuous"
        response_columns_uncoded = []

        # Plot histogram
        resp_plot = plotly_express.histogram(response_columns)
        file_name = "graphs/continious_response.html"
        resp_plot.write_html(file=file_name, include_plotlyjs="cdn")

    # Get response mean
    response_mean = response_columns.mean()

    return response_columns, response_type, response_mean, response_columns_uncoded


def process_predictors(
    data_frames,
    predictor,
    response,
    response_columns,
    response_type,
    response_mean,
    response_columns_uncoded,
):
    # Fetch predictor columns from the data frames
    predictor_columns = predictor
    print("+++++process_predictors++++++++++++")
    print(predictor_columns)

    # Generate a dummy result table with expected results
    results_columns = [
        "Response",
        "Predictor Type",
        "t Score",
        "p Value",
        "Regression Plot",
        "Diff Mean of Response (Unweighted)",
        "Diff Mean of Response (Weighted)",
        "Diff Mean Plot",
    ]

    # Create results from the dataframe using the result columns and predictor
    results = pd.DataFrame(columns=results_columns, index=predictor)
    #print(results)

    # Loop over predictors
    for prediction_name, predictor_data in predictor_columns.items():
        #print("++++NAME+++++++++++++")
        #print(prediction_name)

        # Decide cat or cont
        predictor_type = continuous_or_categorical_predictor(predictor_data)
        #print(predictor_type)
        if predictor_type == "Categorical":
            # Encode
            predictor_data = pd.Categorical(
                predictor_data, categories=predictor_data.unique()
            )
            predictor_data, pred_labels = pd.factorize(predictor_data)
            predictor_data = pd.DataFrame(predictor_data, columns=[prediction_name])
            predictor_data_uncoded = data_frames[prediction_name]
            #print("After processing")
            #print(predictor_data)
        else:
            predictor_data = predictor_data.astype(float)
            predictor_data = predictor_data.to_frame()
            #print("After processing")
            #print(predictor_data)

        # Bind response and predictor together again
        data_frames_c = pd.concat([response_columns, predictor_data], axis=1)
        data_frames_c.columns = [response, prediction_name]

        # Relationship plot and correlations
        #print("response_type")
        #print(response_type)
        #print("predictor_type")
        #print(predictor_type)

        if response_type == "Categorical" and predictor_type == "Categorical":
            relationship_matrix = confusion_matrix(predictor_data, response_columns)
            figure_relationship = go.Figure(
                data=go.Heatmap(
                    z=relationship_matrix, zmin=0, zmax=relationship_matrix.max()
                )
            )
            figure_relationship.update_layout(
                title=f"Relationship Between {response} and {prediction_name}",
                xaxis_title=prediction_name,
                yaxis_title=response,
            )
        elif response_type == "Categorical" and predictor_type == "Continuous":
            figure_relationship = plotly_express.histogram(
                data_frames_c, x=prediction_name, color=response_columns_uncoded
            )
            figure_relationship.update_layout(
                title=f"Relationship Between {response} and {prediction_name}",
                xaxis_title=prediction_name,
                yaxis_title="count",
            )
        elif response_type == "Continuous" and predictor_type == "Categorical":
            figure_relationship = plotly_express.histogram(
                data_frames_c, x=response, color=predictor_data_uncoded
            )
            figure_relationship.update_layout(
                title=f"Relationship Between {response} and {prediction_name}",
                xaxis_title=response,
                yaxis_title="count",
            )
        elif response_type == "Continuous" and predictor_type == "Continuous":
            figure_relationship = plotly_express.scatter(
                y=response_columns, x=predictor_data, trendline="ols"
            )
            figure_relationship.update_layout(
                title=f"Relationship Between {response} and {prediction_name}",
                xaxis_title=prediction_name,
                yaxis_title=response,
            )
        # Remove the blank spaces
        response_html = response.replace(" ", "")
        prediction_name_html = prediction_name.replace(" ", "")

        relationship_file_save = (
            f"./graphs/{response_html}_{prediction_name_html}_relationship.html"
        )
        relationship_file_open = (
            f"./{response_html}_{prediction_name_html}_relationship.html"
        )
        figure_relationship.write_html(
            file=relationship_file_save, include_plotlyjs="cdn"
        )
        relationship_link = (
            "<a target='blank' href="
            + relationship_file_open
            + "><div>"
            + predictor_type
            + "</div></a>"
        )

        # Regression
        print("Regression------------>")
        print(response_type)
        #print(response_columns)
        #print(predictor_data)
        if response_type == "Categorical":
            regression_model = sm.Logit(
                response_columns, predictor_data, missing="drop"
            )
        else:
            regression_model = sm.OLS(response_columns, predictor_data, missing="drop")

        print("Chetan0")
        # Fit model
        regression_model_fitted = regression_model.fit()

        # Get t val and p score
        t_score = round(regression_model_fitted.tvalues[0], 6)
        p_value = "{:.6e}".format(regression_model_fitted.pvalues[0])

        print("Chetan1")
        # Plot regression
        regression_fig = plotly_express.scatter(
            y=data_frames_c[response], x=data_frames_c[prediction_name], trendline="ols"
        )
        regression_fig.write_html(
            file=f"./graphs/{prediction_name}_regression.html", include_plotlyjs="cdn"
        )
        regression_fig.update_layout(
            title=f"Regression: {response} on {prediction_name}",
            xaxis_title=prediction_name,
            yaxis_title=response,
        )

        regression_file_save = (
            f"./graphs/{response_html}_{prediction_name_html}_reg.html"
        )
        regression_file_open = f"./{response_html}_{prediction_name_html}_reg.html"
        regression_fig.write_html(file=regression_file_save, include_plotlyjs="cdn")
        regression_link = (
            "<a target='blank' href=" + regression_file_open + "><div>Plot</div></a>"
        )

        print("Chetan2")
        # Diff with mean of response (unweighted and weighted)
        # Get user input on number of mean diff bins to use
        if predictor_type == "Continuous":
            bin_n = "3"
            while isinstance(bin_n, int) is False or bin_n == "":
                # bin_n = input(
                #    f"\nEnter number of bins to use for difference with mean of response for {prediction_name}:\n"
                # )
                try:
                    bin_n = int(bin_n)
                except Exception:
                    continue
            else:
                pass
            data_frames_c["bin_labels"] = pd.cut(
                data_frames_c[prediction_name], bins=bin_n, labels=False
            )
            binned_means = data_frames_c.groupby("bin_labels").agg(
                {response: ["mean", "count"], prediction_name: "mean"}
            )

        else:
            data_frames_c.columns = [f"{response}", f"{prediction_name}"]
            binned_means = data_frames_c.groupby(predictor_data.iloc[:, 0]).agg(
                {response: ["mean", "count"], prediction_name: "mean"}
            )
            bin_n = len(np.unique(predictor_data.iloc[:, 0].values))

        print("Chetan3")
        binned_means.columns = [f"{response} mean", "count", f"{prediction_name} mean"]

        # Binning and mean squared difference calc
        binned_means["weight"] = binned_means["count"] / binned_means["count"].sum()
        binned_means["mean_sq_diff"] = (
            binned_means[f"{response} mean"].subtract(response_mean, fill_value=0) ** 2
        )
        binned_means["mean_sq_diff_w"] = (
            binned_means["weight"] * binned_means["mean_sq_diff"]
        )

        print("Chetan4")
        # Diff with mean of response stat calculations (weighted and unweighted)
        mean_unweighted = binned_means["mean_sq_diff"].sum() * (1 / bin_n)
        mean_weighted = binned_means["mean_sq_diff_w"].sum()

        # Diff with mean of response plots
        figure_diff = make_subplots(specs=[[{"secondary_y": True}]])
        figure_diff.add_trace(
            go.Bar(
                x=binned_means[f"{prediction_name} mean"],
                y=binned_means["count"],
                name="Observations",
            )
        )
        figure_diff.add_trace(
            go.Scatter(
                x=binned_means[f"{prediction_name} mean"],
                y=binned_means[f"{response} mean"],
                line=dict(color="red"),
                name=f"Relationship with {response}",
            ),
            secondary_y=True,
        )
        figure_diff.update_layout(
            title_text=f"Difference in Mean Response: {response} and {prediction_name}",
        )
        figure_diff.update_xaxes(title_text=f"{prediction_name} (binned)")
        figure_diff.update_yaxes(title_text="count", secondary_y=False)
        figure_diff.update_yaxes(title_text=f"{response}", secondary_y=True)

        figure_diff_file_save = (
            f"./graphs/{response_html}_{prediction_name_html}_diff.html"
        )
        figure_diff_file_open = f"./{response_html}_{prediction_name_html}_diff.html"
        figure_diff.write_html(file=figure_diff_file_save, include_plotlyjs="cdn")
        mean_diff = (
            "<a target='blank' href=" + figure_diff_file_open + "><div>Plot</div></a>"
        )

        print("Chetan5")
        # Create processed data_frames
        if prediction_name == predictor_columns.columns[0]:
            processed_predictor = pd.concat([response_columns, predictor_data], axis=1)
        else:
            processed_predictor = pd.concat(
                [processed_predictor, predictor_data], axis=1
            )

        print("Chetan6")
        # Add to results table
        results.loc[prediction_name] = pd.Series(
            {
                "Response": response,
                "Predictor Type": relationship_link,
                "t Score": t_score,
                "p Value": p_value,
                "Regression Plot": regression_link,
                "Diff Mean of Response (Unweighted)": mean_unweighted,
                "Diff Mean of Response (Weighted)": mean_weighted,
                "Diff Mean Plot": mean_diff,
            }
        )

    return processed_predictor, results


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
    #print("\nresults:\n", results)
    #print(importance)
    results.reset_index(inplace=True, drop=True)
    results = pd.concat([results, importance], axis=1)
    #print("\nresults:\n", results)

    with open("./graphs/results.html", "w") as html_open:
        results.to_html(html_open, escape=False)

    return


def main():
    # https://medium.com/@debanjana.bhattacharyya9818/numpy-random-seed-101-explained-2e96ee3fd90b
    # Seed the generator
    np.random.seed(seed=123)

    # Load the file and fetch response feature
    data_frames, response, predictor = load_file()

    # process response
    (
        response_columns,
        response_type,
        response_mean,
        response_columns_uncoded,
    ) = process_response(data_frames, response)

    (processed_predictor, results) = process_predictors(
        data_frames,
        predictor,
        response,
        response_columns,
        response_type,
        response_mean,
        response_columns_uncoded,
    )

    importance = random_forest_importance(response_type, processed_predictor, predictor)

    results_table(results, importance)
    return


if __name__ == "__main__":
    sys.exit(main())
