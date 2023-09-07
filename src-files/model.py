import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


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
        model.feature_importances_, columns=["Random Forest Importance"]
    )

    return feature_importance


def results_table(
    results_response_x_predictor,
    results_brute_force,
    results_predictor_correlation,
    one_way_importance,
):
    with open("./plots/importance.html", "w") as html_open:
        one_way_importance.to_html(html_open, escape=False)

    results_response_x_predictor.reset_index(inplace=True, drop=True)
    results_response_x_predictor = pd.concat(
        [results_response_x_predictor, one_way_importance], axis=1
    )

    results_response_x_predictor = results_response_x_predictor.sort_values(
        ["Correlation"], ascending=False
    )

    with open("./plots/results_response_x_predictor.html", "w") as html_open:
        results_response_x_predictor.to_html(html_open, escape=False)

    with open("./plots/results_brute_force.html", "w") as html_open:
        results_brute_force.to_html(html_open, escape=False)

    with open("./plots/results_predictor_correlation.html", "w") as html_open:
        results_predictor_correlation.to_html(html_open, escape=False)

    return
