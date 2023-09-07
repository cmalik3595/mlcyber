from itertools import combinations, combinations_with_replacement, product

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from correlation import (
    categorical_categorical_correlation_ratio,
    continious_correlation_ratio,
)
from pandas.api.types import is_string_dtype
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.metrics import confusion_matrix

# import warnings
# warnings.simplefilter(action="ignore", category=FutureWarning)

# Exception handling for chained_assignment warning
# https://stackoverflow.com/questions/47182183/pandas-chained-assignment-warning-exception-handling
pd.options.mode.chained_assignment = None


# Decision rules for categorical:
# - If string
# - If unique values make up less than 5% of total obs
def continuous_or_categorical_result(data_frames, response_columns):
    # Replace NAs with 0s
    response_columns.fillna(0, inplace=True)

    resp_string_check = is_string_dtype(response_columns)
    resp_unique_ratio = len(np.unique(response_columns.values)) / len(
        response_columns.values
    )

    if resp_string_check or resp_unique_ratio < 0.00001:
        return "Categorical"
    else:
        return "Continuous"


def continuous_or_categorical_predictor(predictor_data):
    predictor_string_check = is_string_dtype(predictor_data)
    predictor_unique_ratio = len(predictor_data.unique()) / len(predictor_data)
    if predictor_string_check or predictor_unique_ratio < 0.00001:
        return "Categorical"
    else:
        return "Continuous"


def process_response(data_frames, response):
    # Check var type of response
    response_columns = data_frames[response]
    response_var_type = continuous_or_categorical_result(data_frames, response_columns)

    if response_var_type == "Categorical":
        response_type = "Categorical"

        # Plot histogram
        # resp_col_plot = response_columns.to_frame()
        resp_plot = px.histogram(response_columns)
        file_name = "./plots/response.html"
        resp_plot.write_html(file=file_name, include_plotlyjs="cdn")

        # Encode
        response_columns = pd.Categorical(
            response_columns, categories=response_columns.unique()
        )
        response_columns, resp_labels = pd.factorize(response_columns)

        response_columns = pd.DataFrame(response_columns, columns=[response])
        response_columns_uncoded = data_frames[response]

    else:
        response_type = "Continuous"
        response_columns_uncoded = []

        # Plot histogram
        resp_plot = px.histogram(response_columns)
        file_name = "./plots/response.html"
        resp_plot.write_html(file=file_name, include_plotlyjs="cdn")

    # Get response mean
    response_mean = response_columns.mean()

    return response_columns, response_type, response_mean, response_columns_uncoded


def process_predictors(
    data_frames,
    predicts,
    response,
    response_columns,
    response_type,
    response_mean,
    response_columns_uncoded,
    num_bins,
):

    predicts_columns = predicts
    # predicts_columns = data_frames[data_frames.columns.intersection(predicts)]
    predictor_proc = pd.DataFrame()

    # Build preliminary results table
    results_response_x_predictor_columns = [
        "Response",
        "Predictor name",
        "Predictor Type",
        "Correlation",
        "t Score",
        "p Value",
        "Regression Plot",
        "Diff Mean of Response (Unweighted)",
        "Diff Mean of Response (Weighted)",
        "Diff Mean Plot",
    ]
    results_response_x_predictor = pd.DataFrame(
        columns=results_response_x_predictor_columns
    )
    #    results_response_x_predictor = pd.DataFrame(
    #        columns=results_response_x_predictor_columns, index=predicts
    #    )

    print("\nSelected bins:\n", num_bins)

    # Get user input on bins to use for diff of mean of response
    while num_bins == "0" or num_bins == "":
        print("Error: Invalid bins for mean!!!")
        num_bins = input(
            "\nEnter number of bins to use for difference with mean of response:\n"
        )
        try:
            num_bins = int(num_bins)
        except Exception:
            continue
    else:
        pass
    num_bins = int(num_bins)

    for predictor_name, predictor_data in predicts_columns.iteritems():

        # Replace NAs with 0s
        predictor_data.fillna(0, inplace=True)

        # Decide cat or cont
        predictor_type = continuous_or_categorical_predictor(predictor_data)
        if predictor_type == "Categorical":
            predictor_type = "Categorical"

            # Encode
            predictor_data = pd.Categorical(
                predictor_data, categories=predictor_data.unique()
            )
            predictor_data, predictor_labels = pd.factorize(predictor_data)

            predictor_data = pd.DataFrame(predictor_data, columns=[predictor_name])
            predictor_data_uncoded = data_frames[predictor_name].fillna(0, inplace=True)

        else:
            predictor_type = "Continuous"

        # Bind response and predictor together again
        data_frames_c = pd.concat([response_columns, predictor_data], axis=1)
        data_frames_c.columns = [response, predictor_name]

        # Relationship plot and correlations
        if response_type == "Categorical" and predictor_type == "Categorical":
            relationship_matrix = confusion_matrix(predictor_data, response_columns)
            figure_relate = go.Figure(
                data=go.Heatmap(
                    z=relationship_matrix, zmin=0, zmax=relationship_matrix.max()
                )
            )
            figure_relate.update_layout(
                title=f"Relationship Between {response} and {predictor_name}",
                xaxis_title=predictor_name,
                yaxis_title=response,
            )

            corr = categorical_categorical_correlation_ratio(
                data_frames_c[predictor_name], data_frames_c[response]
            )

        elif response_type == "Categorical" and predictor_type == "Continuous":

            figure_relate = px.histogram(
                data_frames_c, x=predictor_name, color=response_columns_uncoded
            )
            figure_relate.update_layout(
                title=f"Relationship Between {response} and {predictor_name}",
                xaxis_title=predictor_name,
                yaxis_title="count",
            )

            corr = stats.pointbiserialr(
                data_frames_c[response], data_frames_c[predictor_name]
            )[0]

        elif response_type == "Continuous" and predictor_type == "Categorical":

            figure_relate = px.histogram(
                data_frames_c, x=response, color=predictor_data_uncoded
            )
            figure_relate.update_layout(
                title=f"Relationship Between {response} and {predictor_name}",
                xaxis_title=response,
                yaxis_title="count",
            )

            corr = continious_correlation_ratio(
                data_frames_c[predictor_name], data_frames_c[response]
            )

        elif response_type == "Continuous" and predictor_type == "Continuous":

            figure_relate = px.scatter(
                y=response_columns, x=predictor_data, trendline="ols"
            )
            figure_relate.update_layout(
                title=f"Relationship Between {response} and {predictor_name}",
                xaxis_title=predictor_name,
                yaxis_title=response,
            )

            corr = data_frames_c[response].corr(data_frames_c[predictor_name])

        response_html = response.replace(" ", "")
        predictor_name_html = predictor_name.replace(" ", "")

        relate_file_save = f"./plots/{response_html}_{predictor_name_html}_relate.html"
        relate_file_open = f"./{response_html}_{predictor_name_html}_relate.html"
        figure_relate.write_html(file=relate_file_save, include_plotlyjs="cdn")
        relate_link = (
            "<a target='blank' href="
            + relate_file_open
            + "><div>"
            + predictor_type
            + "</div></a>"
        )

        # Regression
        if response_type == "Categorical":
            reg_model = sm.Logit(response_columns, predictor_data, missing="drop")

        else:
            reg_model = sm.OLS(response_columns, predictor_data, missing="drop")

        # Fit model
        reg_model_fitted = reg_model.fit()

        # Get t val and p score
        t_score = round(reg_model_fitted.tvalues[0], 6)
        p_value = "{:.6e}".format(reg_model_fitted.pvalues[0])

        # Plot regression
        reg_fig = px.scatter(
            y=data_frames_c[response], x=data_frames_c[predictor_name], trendline="ols"
        )
        reg_fig.write_html(
            file=f"./plots/{predictor_name}_regression.html", include_plotlyjs="cdn"
        )
        reg_fig.update_layout(
            title=f"Regression: {response} on {predictor_name}",
            xaxis_title=predictor_name,
            yaxis_title=response,
        )

        reg_file_save = f"./plots/{response_html}_{predictor_name_html}_reg.html"
        reg_file_open = f"./{response_html}_{predictor_name_html}_reg.html"
        reg_fig.write_html(file=reg_file_save, include_plotlyjs="cdn")
        reg_link = "<a target='blank' href=" + reg_file_open + "><div>Plot</div></a>"

        # Diff with mean of response (unweighted and weighted)
        if predictor_type == "Continuous":
            data_frames_c["bin_labels"] = pd.cut(
                data_frames_c[predictor_name], bins=num_bins, labels=False
            )
            binned_means = data_frames_c.groupby("bin_labels").agg(
                {response: ["mean", "count"], predictor_name: "mean"}
            )
            useful_bins = num_bins

        else:
            data_frames_c.columns = [f"{response}", f"{predictor_name}"]
            binned_means = data_frames_c.groupby(predictor_data.iloc[:, 0]).agg(
                {response: ["mean", "count"], predictor_name: "mean"}
            )
            useful_bins = len(np.unique(predictor_data.iloc[:, 0].values))

        binned_means.columns = [f"{response} mean", "count", f"{predictor_name} mean"]

        # Binning and mean squared difference calc
        binned_means["weight"] = binned_means["count"] / binned_means["count"].sum()
        binned_means["mean_sq_diff"] = (
            binned_means[f"{response} mean"].subtract(response_mean, fill_value=0) ** 2
        )
        binned_means["mean_sq_diff_w"] = (
            binned_means["weight"] * binned_means["mean_sq_diff"]
        )

        # Diff with mean of response stat calculations (weighted and unweighted)
        mean_unweighted = binned_means["mean_sq_diff"].sum() * (1 / useful_bins)
        mean_weighted = binned_means["mean_sq_diff_w"].sum()

        # Diff with mean of response plots
        figure_diff = make_subplots(specs=[[{"secondary_y": True}]])
        figure_diff.add_trace(
            go.Bar(
                x=binned_means[f"{predictor_name} mean"],
                y=binned_means["count"],
                name="Observations",
            )
        )
        figure_diff.add_trace(
            go.Scatter(
                x=binned_means[f"{predictor_name} mean"],
                y=binned_means[f"{response} mean"],
                line=dict(color="red"),
                name=f"Relationship with {response}",
            ),
            secondary_y=True,
        )
        figure_diff.update_layout(
            title_text=f"Difference in Mean Response: {response} and {predictor_name}",
        )
        figure_diff.update_xaxes(title_text=f"{predictor_name} (binned)")
        figure_diff.update_yaxes(title_text="count", secondary_y=False)
        figure_diff.update_yaxes(title_text=f"{response}", secondary_y=True)

        figure_diff_file_save = (
            f"./plots/{response_html}_{predictor_name_html}_diff.html"
        )
        figure_diff_file_open = f"./{response_html}_{predictor_name_html}_diff.html"
        figure_diff.write_html(file=figure_diff_file_save, include_plotlyjs="cdn")
        diff_link = (
            "<a target='blank' href=" + figure_diff_file_open + "><div>Plot</div></a>"
        )

        # Create processed data_frames
        if predictor_name == predicts_columns.columns[0]:
            predictor_proc = pd.concat([response_columns, predictor_data], axis=1)
        else:
            predictor_proc = pd.concat([predictor_proc, predictor_data], axis=1)

        # Add to results (response_x_predictor) table
        # results_response_x_predictor.loc[predictor_name] = pd.Series(
        # {
        # "Response": response,
        # "Predictor name": prediction_name,
        # "Predictor Type": relate_link,
        # "Correlation": corr,
        # "t Score": t_score,
        # "p Value": p_value,
        # "Regression Plot": reg_link,
        # "Diff Mean of Response (Unweighted)": mean_unweighted,
        # "Diff Mean of Response (Weighted)": mean_weighted,
        # "Diff Mean Plot": diff_link,
        # }
        # )
        series = pd.Series(
            {
                "Response": response,
                "Predictor name": predictor_name,
                "Predictor Type": relate_link,
                "Correlation": corr,
                "t Score": t_score,
                "p Value": p_value,
                "Regression Plot": reg_link,
                "Diff Mean of Response (Unweighted)": mean_unweighted,
                "Diff Mean of Response (Weighted)": mean_weighted,
                "Diff Mean Plot": diff_link,
            },
            name=predictor_name,
        )
        results_response_x_predictor = results_response_x_predictor.append(series)

    # results_response_x_predictor = results_response_x_predictor.sort_values(
    #    ["Correlation"], ascending=False
    # )

    return predictor_proc, results_response_x_predictor, predicts_columns, num_bins


def process_predictors_two_way(
    response, predicts_columns, num_bins, response_columns, response_mean
):
    combination_values = list(combinations(predicts_columns.columns, 2))

    combinations_len = range(1, len(combination_values))

    # Build preliminary results table - brute force
    results_brute_force_cols = [
        "Response",
        "Predictor 1",
        "Predictor 2",
        "Predictor 1 Type",
        "Predictor 2 Type",
        "DMR Unweighted",
        "DMR Weighted",
        "DMR Weighted Plot",
    ]
    results_brute_force = pd.DataFrame(
        columns=results_brute_force_cols, index=combinations_len
    )

    # Build preliminary results table - correlation table
    results_predictor_correlation_cols = [
        "Response",
        "Predictor 1",
        "Predictor 2",
        "Predictor 1 Type",
        "Resp/Pred 1 Plot",
        "Predictor 2 Type",
        "Resp/Pred 2 Plot",
        "Correlation",
    ]
    results_predictor_correlation = pd.DataFrame(
        columns=results_predictor_correlation_cols, index=combinations_len
    )

    combination_pos = 1

    for combination in combination_values:

        predictor1_name = combination[0]
        predictor2_name = combination[1]

        predictor_data_1 = predicts_columns[combination[0]]
        predictor_data_2 = predicts_columns[combination[1]]

        # Decide cat or cont
        predictor1_type = continuous_or_categorical_predictor(predictor_data_1)
        if predictor1_type == "Categorical":
            predictor1_type = "Categorical"

            # Encode
            predictor_data_1 = pd.Categorical(
                predictor_data_1, categories=predictor_data_1.unique()
            )
            predictor_data_1, predictor_labels_1 = pd.factorize(predictor_data_1)

            predictor_data_1 = pd.DataFrame(predictor_data_1, columns=[predictor1_name])

        else:
            predictor1_type = "Continuous"

        # Decide cat or cont
        predictor2_type = continuous_or_categorical_predictor(predictor_data_2)
        if predictor2_type == "Categorical":
            predictor2_type = "Categorical"

            # Encode
            predictor_data_2 = pd.Categorical(
                predictor_data_2, categories=predictor_data_2.unique()
            )
            predictor_data_2, predictor_labels_2 = pd.factorize(predictor_data_2)

            predictor_data_2 = pd.DataFrame(predictor_data_2, columns=[predictor2_name])

        else:
            predictor2_type = "Continuous"

        # Bind response and predictors
        data_frames_p = pd.concat(
            [response_columns, predictor_data_1, predictor_data_2], axis=1
        )

        if predictor1_type == "Categorical" and predictor2_type == "Categorical":
            corr = categorical_categorical_correlation_ratio(
                data_frames_p[predictor2_name], data_frames_p[predictor1_name]
            )

        elif (
            predictor1_type == "Categorical"
            and predictor2_type == "Continuous"
            or predictor1_type == "Continuous"
            and predictor2_type == "Categorical"
        ):

            if predictor1_type == "Categorical":

                corr = continious_correlation_ratio(
                    data_frames_p[predictor1_name], data_frames_p[predictor2_name]
                )

            elif predictor2_type == "Categorical":

                corr = continious_correlation_ratio(
                    data_frames_p[predictor2_name], data_frames_p[predictor1_name]
                )

        elif predictor1_type == "Continuous" and predictor2_type == "Continuous":

            corr = data_frames_p[predictor1_name].corr(data_frames_p[predictor2_name])

        # Mean of response two-way calculation
        if predictor1_type == "Continuous":
            data_frames_p["bin_labels_1"] = pd.cut(
                data_frames_p[predictor1_name], bins=num_bins, labels=False
            )
            bin_1_f = num_bins

        else:
            data_frames_p["bin_labels_1"] = data_frames_p[predictor1_name]
            bin_1_f = len(np.unique(predictor_data_1.iloc[:, 0].values))

        if predictor2_type == "Continuous":
            data_frames_p["bin_labels_2"] = pd.cut(
                data_frames_p[predictor2_name], bins=num_bins, labels=False
            )
            bin_2_f = num_bins

        else:
            data_frames_p["bin_labels_2"] = data_frames_p[predictor2_name]
            bin_2_f = len(np.unique(predictor_data_2.iloc[:, 0].values))

        binned_means_total = data_frames_p.groupby(
            ["bin_labels_1", "bin_labels_2"], as_index=False
        ).agg({response: ["mean", "count"]})

        squared_diff = (
            binned_means_total.iloc[
                :, binned_means_total.columns.get_level_values(1) == "mean"
            ].sub(response_mean, level=0)
            ** 2
        )

        binned_means_total["mean_sq_diff"] = squared_diff

        weights_group = binned_means_total.iloc[
            :, binned_means_total.columns.get_level_values(1) == "count"
        ]

        weights_tot = binned_means_total.iloc[
            :, binned_means_total.columns.get_level_values(1) == "count"
        ].sum()

        binned_means_total["weight"] = weights_group.div(weights_tot)

        binned_means_total["mean_sq_diff_w"] = (
            binned_means_total["weight"] * binned_means_total["mean_sq_diff"]
        )

        plot_data = binned_means_total.pivot(
            index="bin_labels_1", columns="bin_labels_2", values="mean_sq_diff_w"
        )
        figure_dmr = go.Figure(data=[go.Surface(z=plot_data.values)])
        figure_dmr.update_layout(
            title=f"DMR (Weighted): {predictor1_name} and {predictor2_name}",
            autosize=True,
            scene=dict(
                xaxis_title=predictor1_name,
                yaxis_title=predictor2_name,
                zaxis_title=response,
            ),
        )

        mean_unweighted_group = binned_means_total["mean_sq_diff"].sum() * (
            1 / (bin_1_f * bin_2_f)
        )
        mean_weighted_group = binned_means_total["mean_sq_diff_w"].sum()

        predictor1_name_html = predictor1_name.replace(" ", "")
        predictor2_name_html = predictor2_name.replace(" ", "")

        figure_dmr_file_save = (
            f"./plots/{predictor1_name_html}_{predictor2_name_html}_dmr.html"
        )
        figure_dmr_file_open = (
            f"./{predictor1_name_html}_{predictor2_name_html}_dmr.html"
        )
        figure_dmr.write_html(file=figure_dmr_file_save, include_plotlyjs="cdn")
        figure_dmr_link = (
            "<a target='blank' href=" + figure_dmr_file_open + "><div>Plot</div></a>"
        )

        # Add in relationship plot links
        response_html = response.replace(" ", "")

        relate_file_open_1 = f"./{response_html}_{predictor1_name_html}_relate.html"
        relate_link1 = (
            "<a target='blank' href=" + relate_file_open_1 + "><div>Plot</div></a>"
        )

        relate_file_open_2 = f"./{response_html}_{predictor2_name_html}_relate.html"
        relate_link2 = (
            "<a target='blank' href=" + relate_file_open_2 + "><div>Plot</div></a>"
        )

        results_brute_force.loc[combination_pos] = pd.Series(
            {
                "Response": response,
                "Predictor 1": predictor1_name,
                "Predictor 2": predictor2_name,
                "Predictor 1 Type": predictor1_type,
                "Predictor 2 Type": predictor2_type,
                "DMR Unweighted": mean_unweighted_group,
                "DMR Weighted": mean_weighted_group,
                "DMR Weighted Plot": figure_dmr_link,
            }
        )

        results_predictor_correlation.loc[combination_pos] = pd.Series(
            {
                "Response": response,
                "Predictor 1": predictor1_name,
                "Predictor 2": predictor2_name,
                "Predictor 1 Type": predictor1_type,
                "Resp/Pred 1 Plot": relate_link1,
                "Predictor 2 Type": predictor2_type,
                "Resp/Pred 2 Plot": relate_link2,
                "Correlation": corr,
            }
        )

        combination_pos += 1

    results_brute_force = results_brute_force.sort_values(
        ["DMR Weighted"], ascending=False
    )

    results_predictor_correlation = results_predictor_correlation.sort_values(
        ["Correlation"], ascending=False
    )

    return results_brute_force, results_predictor_correlation


def correlation_matrix(results_predictor_correlation, predicts_columns):

    types_data_frames_1 = results_predictor_correlation[
        ["Predictor 1", "Predictor 1 Type"]
    ]
    types_data_frames_1.columns = ["Predictor", "Type"]

    types_data_frames_2 = results_predictor_correlation[
        ["Predictor 2", "Predictor 2 Type"]
    ]
    types_data_frames_2.columns = ["Predictor", "Type"]

    types_data_frames = types_data_frames_1.append(types_data_frames_2)

    types = np.unique(types_data_frames["Type"])

    type_combinations = list(combinations_with_replacement(types, 2))

    for t_combination in type_combinations:

        var_type_1 = t_combination[0]
        var_type_2 = t_combination[1]

        var_names_1 = types_data_frames.loc[
            types_data_frames["Type"] == var_type_1, "Predictor"
        ].unique()
        var_names_2 = types_data_frames.loc[
            types_data_frames["Type"] == var_type_2, "Predictor"
        ].unique()

        var_data_frames_1 = predicts_columns[var_names_1]
        var_data_frames_2 = predicts_columns[var_names_2]

        if var_type_1 == var_type_2 == "Continuous":

            correlation_continious_matrix = var_data_frames_1.corr()

            continious_continious_matrix = px.imshow(
                correlation_continious_matrix,
                labels=dict(color="Pearson correlation:"),
                title=f"Correlation Matrix: {var_type_1} vs {var_type_2}",
            )
            continious_continious_matrix_save = (
                "./plots/continious_continious_matrix.html"
            )
            continious_continious_matrix.write_html(
                file=continious_continious_matrix_save, include_plotlyjs="cdn"
            )

        elif var_type_1 == var_type_2 == "Categorical":

            var_factorized = var_data_frames_1.apply(lambda x: pd.factorize(x)[0])

            categorical_combinations = list(product(var_factorized.columns, repeat=2))
            categorical_combinations_len = range(0, len(categorical_combinations))

            categorical_correlation_columns = [
                "Predictor 1",
                "Predictor 2",
                "Correlation",
            ]
            categorical_corr = pd.DataFrame(
                columns=categorical_correlation_columns,
                index=categorical_combinations_len,
            )

            categorical_pos = 0

            for categorical_combination in categorical_combinations:

                categorical_name_1 = categorical_combination[0]
                categorical_name_2 = categorical_combination[1]

                corr = categorical_categorical_correlation_ratio(
                    var_factorized[categorical_name_1],
                    var_factorized[categorical_name_2],
                )

                categorical_corr.loc[categorical_pos] = pd.Series(
                    {
                        "Predictor 1": categorical_name_1,
                        "Predictor 2": categorical_name_2,
                        "Correlation": corr,
                    }
                )

                categorical_pos += 1

            correlation_categorical_matrix = categorical_corr.pivot(
                index="Predictor 1", columns="Predictor 2", values="Correlation"
            )

            categorical_categorical_matrix = px.imshow(
                correlation_categorical_matrix,
                labels=dict(color="Cramer's V"),
                title=f"Correlation Matrix: {var_type_1} vs {var_type_2}",
            )
            categorical_categorical_matrix_save = (
                "./plots/categorical_categorical_matrix.html"
            )
            categorical_categorical_matrix.write_html(
                file=categorical_categorical_matrix_save, include_plotlyjs="cdn"
            )

        elif (
            var_type_1 == "Categorical"
            and var_type_2 == "Continuous"
            or var_type_1 == "Continuous"
            and var_type_2 == "Categorical"
        ):

            categorical_continious_combinations = list(
                product(var_names_1, var_names_2)
            )
            categorical_continious_combinations_len = range(
                0, len(categorical_continious_combinations)
            )

            categorical_continious_correlation_cols = [
                "Predictor 1",
                "Predictor 2",
                "Correlation",
            ]
            categorical_continious_corr = pd.DataFrame(
                columns=categorical_continious_correlation_cols,
                index=categorical_continious_combinations_len,
            )

            categorical_continious_pos = 0

            for (
                categorical_continious_combination
            ) in categorical_continious_combinations:

                categorical_continious_name_1 = categorical_continious_combination[0]
                categorical_continious_name_2 = categorical_continious_combination[1]

                if var_type_1 == "Categorical":

                    corr = continious_correlation_ratio(
                        var_data_frames_1[categorical_continious_name_1],
                        var_data_frames_2[categorical_continious_name_2],
                    )

                elif var_type_2 == "Categorical":

                    corr = continious_correlation_ratio(
                        var_data_frames_2[categorical_continious_name_2],
                        var_data_frames_1[categorical_continious_name_1],
                    )

                categorical_continious_corr.loc[categorical_continious_pos] = pd.Series(
                    {
                        "Predictor 1": categorical_continious_name_1,
                        "Predictor 2": categorical_continious_name_2,
                        "Correlation": corr,
                    }
                )

                categorical_continious_pos += 1

            correlation_categorical_continious_matrix = (
                categorical_continious_corr.pivot(
                    index="Predictor 1", columns="Predictor 2", values="Correlation"
                )
            )

            categorical_continious_matrix = px.imshow(
                correlation_categorical_continious_matrix,
                labels=dict(color="Correlation Ratio"),
                title=f"Correlation Matrix: {var_type_1} vs {var_type_2}",
            )
            categorical_continious_matrix_save = (
                "./plots/categorical_continious_matrix.html"
            )
            categorical_continious_matrix.write_html(
                file=categorical_continious_matrix_save, include_plotlyjs="cdn"
            )

    return
