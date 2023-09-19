import sys

import pandas as pd
import features
import loaddata
import model
import numpy as np
import pipeline


def main(in_file_name, in_response_type, in_bins):
    # https://medium.com/@debanjana.bhattacharyya9818/numpy-random-seed-101-explained-2e96ee3fd90b
    np.random.seed(seed=1234)
    data_frames, response, predicts = loaddata.load_file()

    (
        response_columns,
        response_type,
        response_mean,
        response_columns_uncoded,
    ) = features.process_response(data_frames, response)
    (
        predictor_proc,
        results_response_x_predictor,
        predicts_columns,
        num_bins,
        results_response_x_predictor_indexed,
    ) = features.process_predictors(
        data_frames,
        predicts,
        response,
        response_columns,
        response_type,
        response_mean,
        response_columns_uncoded,
        in_bins,
    )
    perm_importance = model.perm_importance(
        response_type, predictor_proc
    )

    random_forrest_importance = model.random_forest_importance(
        response_type, predictor_proc, predicts
    )
    
    linear_regression_importance = model.linear_regression_importance(
        response_type, predictor_proc
    )
    
    logistic_regression_importance = model.logistic_regression_importance(
        response_type, predictor_proc
    )

    decision_tree_importance = model.decision_tree_importance(
        response_type, predictor_proc
    )

    xgboost_importance = model.xgboost_importance(
        response_type, predictor_proc
    )

    (
        results_brute_force,
        results_predictor_correlation,
    ) = features.process_predictors_two_way(
        response, predicts_columns, num_bins, response_columns, response_mean
    )

    features.correlation_matrix(results_predictor_correlation, predicts_columns)

    (results_response_x_predictor) = model.results_table(
        results_response_x_predictor,
        results_brute_force,
        results_predictor_correlation,
        random_forrest_importance,
        linear_regression_importance,
        logistic_regression_importance,
        decision_tree_importance,
        xgboost_importance,
        perm_importance,
        results_response_x_predictor_indexed,
    )
    
    print("Before")
    print(predictor_proc.shape)
    with open("./plots/before_trimming.html", "w") as html_open:
        predictor_proc.to_html(html_open, escape=False)

    trimmed_features0 = [predictor_proc.columns]

    pipeline.try_models(
        predictor_proc.iloc[:, 1:],
        pd.Series(predictor_proc.iloc[:, 0]),
        trimmed_features0,
        "./plots/before_trimming",
        "Pipeline",
    )

    (predictor_proc) = model.post_process_data(results_response_x_predictor, predictor_proc)
    
    print("After")
    print(predictor_proc.shape)
    with open("./plots/after_trimming.html", "w") as html_open:
        predictor_proc.to_html(html_open, escape=False)

    trimmed_features1 = [predictor_proc.columns]

    pipeline.try_models(
        predictor_proc.iloc[:, 1:],
        pd.Series(predictor_proc.iloc[:, 0]),
        trimmed_features1,
        "./plots/after_trimming",
        "Pipeline",
    )

    return


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("\nUsage:")
        print("python3 main.py <file_path> <final response name> <number of bins>")
        print("Example: python3 main.py ../data/dataset.csv class3 10")
        sys.exit()

    in_file = sys.argv[1]
    in_response = sys.argv[2]
    in_bins = sys.argv[3]
    sys.exit(main(in_file, in_response, in_bins))
