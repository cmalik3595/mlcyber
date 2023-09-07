import sys

import features
import loaddata
import model
import numpy as np


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
    one_way_importance = model.random_forest_importance(
        response_type, predictor_proc, predicts
    )

    (
        results_brute_force,
        results_predictor_correlation,
    ) = features.process_predictors_two_way(
        response, predicts_columns, num_bins, response_columns, response_mean
    )

    features.correlation_matrix(results_predictor_correlation, predicts_columns)

    model.results_table(
        results_response_x_predictor,
        results_brute_force,
        results_predictor_correlation,
        one_way_importance,
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
