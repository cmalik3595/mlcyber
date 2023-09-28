"""
import xgboost
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor


def perm_importance(response_type, processed_predictor):
    """
    if response_type == "Categorical":
        model = KNeighborsClassifier()
    else:
        model = KNeighborsRegressor()
    # Replace missing values with mean
    na_cols = processed_predictor.columns[processed_predictor.isna().any()].tolist()
    if na_cols == "":
        pass
    else:
        processed_predictor.loc[:, na_cols] = processed_predictor.loc[
            :, na_cols
        ].fillna(processed_predictor.loc[:, na_cols].mean())

    X = processed_predictor.iloc[:, 1:].to_numpy()
    y = processed_predictor.iloc[:, 0].to_numpy()

    # fit the model
    model.fit(processed_predictor.iloc[:, 1:], processed_predictor.iloc[:, 0])

    # perform permutation importance
    if response_type == "Categorical":
        results = permutation_importance(model, X, y, scoring="accuracy")
    else:
        results = permutation_importance(model, X, y, scoring="neg_mean_squared_error")

    # get importance
    importance = results.importances_mean
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))

    feature_importance = pd.DataFrame(
        importance, columns=["Permutation Importance"]
    )
    with open("./plots/importance/permutation_importance.html", "w") as html_open:
        feature_importance.to_html(html_open, escape=False)
    return feature_importance
    """
    return pd.DataFrame()


def xgboost_importance(response_type, processed_predictor):
    if response_type == "Categorical":
        model = XGBClassifier()
    else:
        model = XGBRegressor()
    # Replace missing values with mean
    na_cols = processed_predictor.columns[processed_predictor.isna().any()].tolist()
    if na_cols == "":
        pass
    else:
        processed_predictor.loc[:, na_cols] = processed_predictor.loc[
            :, na_cols
        ].fillna(processed_predictor.loc[:, na_cols].mean())
    # fit the model
    model.fit(processed_predictor.iloc[:, 1:], processed_predictor.iloc[:, 0])
    # get importance
    importance = model.feature_importances_
    # summarize feature importance
    for i, v in enumerate(importance):
        print("Feature: %0d, Score: %.5f" % (i, v))

    feature_importance = pd.DataFrame(importance, columns=["XGBoost Importance"])
    with open("./plots/importance/xgboost_importance.html", "w") as html_open:
        feature_importance.to_html(html_open, escape=False)
    return feature_importance


def decision_tree_importance(response_type, processed_predictor):
    if response_type == "Categorical":
        model = DecisionTreeClassifier()
    else:
        model = DecisionTreeRegressor()
    # Replace missing values with mean
    na_cols = processed_predictor.columns[processed_predictor.isna().any()].tolist()
    if na_cols == "":
        pass
    else:
        processed_predictor.loc[:, na_cols] = processed_predictor.loc[
            :, na_cols
        ].fillna(processed_predictor.loc[:, na_cols].mean())
    # fit the model
    model.fit(processed_predictor.iloc[:, 1:], processed_predictor.iloc[:, 0])
    # get importance
    importance = model.feature_importances_
    # summarize feature importance
    for i, v in enumerate(importance):
        print("Feature: %0d, Score: %.5f" % (i, v))

    feature_importance = pd.DataFrame(importance, columns=["Decision Tree Importance"])
    with open("./plots/importance/decision_tree_importance.html", "w") as html_open:
        feature_importance.to_html(html_open, escape=False)
    return feature_importance


def logistic_regression_importance(response_type, processed_predictor):
    # Replace missing values with mean
    na_cols = processed_predictor.columns[processed_predictor.isna().any()].tolist()
    if na_cols == "":
        pass
    else:
        processed_predictor.loc[:, na_cols] = processed_predictor.loc[
            :, na_cols
        ].fillna(processed_predictor.loc[:, na_cols].mean())
    # define the model
    model = LogisticRegression(max_iter=1500)
    # fit the model
    model.fit(processed_predictor.iloc[:, 1:], processed_predictor.iloc[:, 0])
    # get importance
    importance = model.coef_[0]
    # summarize feature importance
    for i, v in enumerate(importance):
        print("Feature: %0d, Score: %.5f" % (i, v))

    feature_importance = pd.DataFrame(
        importance, columns=["Linear Regression Importance"]
    )
    with open(
        "./plots/importance/logistic_regression_importance.html", "w"
    ) as html_open:
        feature_importance.to_html(html_open, escape=False)
    return feature_importance


def linear_regression_importance(response_type, processed_predictor):
    # Replace missing values with mean
    na_cols = processed_predictor.columns[processed_predictor.isna().any()].tolist()
    if na_cols == "":
        pass
    else:
        processed_predictor.loc[:, na_cols] = processed_predictor.loc[
            :, na_cols
        ].fillna(processed_predictor.loc[:, na_cols].mean())
    # define the model
    model = LinearRegression()
    # fit the model
    model.fit(processed_predictor.iloc[:, 1:], processed_predictor.iloc[:, 0])
    # get importance
    importance = model.coef_
    # summarize feature importance
    for i, v in enumerate(importance):
        print("Feature: %0d, Score: %.5f" % (i, v))
    feature_importance = pd.DataFrame(
        importance, columns=["Linear Regression Importance"]
    )
    with open("./plots/importance/linear_regression_importance.html", "w") as html_open:
        feature_importance.to_html(html_open, escape=False)
    return feature_importance


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
    with open("./plots/importance/random_forrest_importance.html", "w") as html_open:
        feature_importance.to_html(html_open, escape=False)

    return feature_importance


def results_table(
    results_response_x_predictor,
    results_brute_force,
    results_predictor_correlation,
    random_forrest_importance,
    linear_regression_importance,
    logistic_regression_importance,
    decision_tree_importance,
    xgboost_importance,
    perm_imp,
    results_response_x_predictor_indexed,
):

    results_response_x_predictor.reset_index(inplace=True, drop=True)

    results_response_x_predictor = pd.concat(
        [results_response_x_predictor, random_forrest_importance], axis=1
    )
    results_response_x_predictor = pd.concat(
        [results_response_x_predictor, linear_regression_importance], axis=1
    )
    results_response_x_predictor = pd.concat(
        [results_response_x_predictor, logistic_regression_importance], axis=1
    )
    results_response_x_predictor = pd.concat(
        [results_response_x_predictor, decision_tree_importance], axis=1
    )
    results_response_x_predictor = pd.concat(
        [results_response_x_predictor, xgboost_importance], axis=1
    )
    results_response_x_predictor = pd.concat(
        [results_response_x_predictor, perm_imp], axis=1
    )

    results_response_corr_predictor = results_response_x_predictor.sort_values(
        ["Correlation"], ascending=False
    )
    results_response_imp_predictor = results_response_x_predictor.sort_values(
        ["Random Forest Importance"], ascending=False
    )

    with open("./plots/results_response_imp_predictor.html", "w") as html_open:
        results_response_imp_predictor.to_html(html_open, escape=False)

    with open("./plots/results_response_corr_predictor.html", "w") as html_open:
        results_response_corr_predictor.to_html(html_open, escape=False)

    with open("./plots/results_brute_force.html", "w") as html_open:
        results_brute_force.to_html(html_open, escape=False)

    with open("./plots/results_predictor_correlation.html", "w") as html_open:
        results_predictor_correlation.to_html(html_open, escape=False)

    return results_response_x_predictor


def post_process_data(results_response_x_predictor, processed_predictor):
    # Only keep x under threshold
    p_val_threshold = 0.05
    t_score_threshold = 2.0
    correlation_thres = 0.0016
    rf_importance_threshold = 0.0003

    for index, row in results_response_x_predictor.iterrows():
        if float(row["p Value"]) > p_val_threshold:
            print("p value exceeding threshold for : ", row["Predictor name"])
            processed_predictor = processed_predictor.drop(
                row["Predictor name"], axis=1
            )
            continue
        if float(row["t Score"]) < t_score_threshold:
            print("t-score exceeding threshold for : ", row["Predictor name"])
            processed_predictor = processed_predictor.drop(
                row["Predictor name"], axis=1
            )
            continue
        if float(row["Correlation"]) < correlation_thres:
            print("correlation exceeding threshold for : ", row["Predictor name"])
            processed_predictor = processed_predictor.drop(
                row["Predictor name"], axis=1
            )
            continue
        if float(row["Random Forest Importance"]) < rf_importance_threshold:
            print("rf importance exceeding threshold for : ", row["Predictor name"])
            processed_predictor = processed_predictor.drop(
                row["Predictor name"], axis=1
            )
            continue

    return processed_predictor
