import loaddata
import features
import model
import numpy as np
import pandas as pd
import sys

def main():
    # https://medium.com/@debanjana.bhattacharyya9818/numpy-random-seed-101-explained-2e96ee3fd90b
    # Seed the generator
    np.random.seed(seed=123)

    # Load the file and fetch response feature
    data_frames, response, predictor = loaddata.load_file()

    # process response
    (
        response_columns,
        response_type,
        response_mean,
        response_columns_uncoded,
    ) = features.process_response(data_frames, response)

    (processed_predictor, results) = features.process_predictors(
        data_frames,
        predictor,
        response,
        response_columns,
        response_type,
        response_mean,
        response_columns_uncoded,
    )

    importance = model.random_forest_importance(response_type, processed_predictor, predictor)

    model.results_table(results, importance)
    return


if __name__ == "__main__":
    sys.exit(main())
