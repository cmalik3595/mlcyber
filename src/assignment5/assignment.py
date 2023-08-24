import os
import sys

import bins
import data
import metrics
import models
import plot
import report


def load():
    # Load data
    response_series = data.load_responses()
    roll_predictor_list = data.load_rolling_100_pitcher_columns()
    career_predictor_list = data.load_prior_career_pitcher_columns()

    # Map Y for X.
    response_series = response_series.loc[
        response_series.index.isin(roll_predictor_list.index)
    ]

    # same for X with y
    roll_predictor_list = roll_predictor_list.loc[
        roll_predictor_list.index.isin(response_series.index)
    ]
    career_predictor_list = career_predictor_list.loc[
        career_predictor_list.index.isin(response_series.index)
    ]

    # Log this instead, eventually
    print(f"{response_series}\n{roll_predictor_list}\n{career_predictor_list}")

    return response_series, roll_predictor_list, career_predictor_list


def predictor_processing(roll_predictor_list, career_predictor_list, response_series):
    # Plot feature distribution by response
    for _, x in roll_predictor_list.items():
        figure = plot.plot_violin(x, response_series, "100-day rolling")
        plot.save_figure(
            figure,
            f"./output/100-day/plots/{x.name}_feature_plot.html",
        )
    for _, x in career_predictor_list.items():
        figure = plot.plot_violin(x, response_series, "prior career")
        plot.save_figure(
            figure,
            f"./output/career/plots/{x.name}_feature_plot.html",
        )


def main() -> int:
    # Make a folder to store the plots
    if not os.path.exists("output"):
        os.makedirs("output")

    if not os.path.exists("output/model-comparison"):
        os.makedirs("output/model-comparison")

    if not os.path.exists("output/model-comparison/plots"):
        os.makedirs("output/model-comparison/plots")

    if not os.path.exists("output/100-day"):
        os.makedirs("output/100-day")

    if not os.path.exists("output/career"):
        os.makedirs("output/career")
    response_series, roll_predictor_list, career_predictor_list = load()

    predictor_processing(roll_predictor_list, career_predictor_list, response_series)

    # Bin the responses into n=10 populations.
    roll_predictor_list_bins = {
        x.name: bins.bin_predictor(x, response_series)
        for _, x in roll_predictor_list.items()
    }
    career_predictor_list_bins = {
        x.name: bins.bin_predictor(x, response_series)
        for _, x in career_predictor_list.items()
    }
    print(f"{roll_predictor_list_bins}\n{career_predictor_list_bins}\n")

    # Plot weighted and unweighted
    for _, x in roll_predictor_list.items():
        figure = plot.plot_bins(
            x.name, response_series.name, roll_predictor_list_bins[x.name]
        )
        plot.save_figure(
            figure,
            f"./output/100-day/plots/{x.name}_bin_plot.html",
        )
    for _, x in career_predictor_list.items():
        figure = plot.plot_bins(
            x.name, response_series.name, career_predictor_list_bins[x.name]
        )
        plot.save_figure(
            figure,
            f"./output/career/plots/{x.name}_bin_plot.html",
        )

    # Calculate and score metrics for each predictor
    roll_predictor_list_metrics = metrics.calculate_predictor_metrics(
        roll_predictor_list, roll_predictor_list_bins, response_series
    )
    career_predictor_list_metrics = metrics.calculate_predictor_metrics(
        career_predictor_list, career_predictor_list_bins, response_series
    )
    print(f"{roll_predictor_list_metrics}\n{career_predictor_list_metrics}\n")

    # Report predictors
    report.report_predictors(
        roll_predictor_list_metrics,
        "predictor_report.html",
        "./output/100-day/",
    )
    report.report_predictors(
        career_predictor_list_metrics,
        "predictor_report.html",
        "./output/career/",
    )

    # Calculate correlations
    roll_predictor_list_corrs = metrics.calculate_correlations(roll_predictor_list)
    career_predictor_list_corrs = metrics.calculate_correlations(career_predictor_list)
    print(f"{roll_predictor_list_corrs}\n{career_predictor_list_corrs}\n")

    # Plot correlations
    figure = plot.plot_correlations(roll_predictor_list_corrs)
    plot.save_figure(
        figure,
        "./output/100-day/plots/corr_plot.html",
    )
    figure = plot.plot_correlations(career_predictor_list_corrs)
    plot.save_figure(
        figure,
        "./output/career/plots/corr_plot.html",
    )

    # Report correlations
    report.report_correlations(
        roll_predictor_list_corrs,
        "corr_report.html",
        "./output/100-day/",
    )
    report.report_correlations(
        career_predictor_list_corrs,
        "corr_report.html",
        "./output/career/",
    )

    # Calculate brute-force
    roll_predictor_list_bf_bins = bins.bin_combinations(
        roll_predictor_list, response_series, n=5
    )
    career_predictor_list_bf_bins = bins.bin_combinations(
        career_predictor_list, response_series, n=5
    )

    # Plot bfs
    for key, b in roll_predictor_list_bf_bins.items():
        figure = plot.plot_bruteforce(b, key[0], key[1])
        plot.save_figure(
            figure,
            f"./output/100-day/plots/{key[0]}X{key[1]}_bf_plot.html",
        )
    for key, b in career_predictor_list_bf_bins.items():
        figure = plot.plot_bruteforce(b, key[0], key[1])
        plot.save_figure(
            figure,
            f"./output/career/plots/{key[0]}X{key[1]}_bf_plot.html",
        )

    # Rank bfs
    roll_predictor_list_bf_ranks = metrics.rank_bf(roll_predictor_list_bf_bins)
    career_predictor_list_bf_ranks = metrics.rank_bf(career_predictor_list_bf_bins)

    # Report bfs
    report.report_bruteforces(
        roll_predictor_list_bf_ranks,
        "bf_report.html",
        "./output/100-day/",
    )
    report.report_bruteforces(
        career_predictor_list_bf_ranks,
        "bf_report.html",
        "./output/career/",
    )

    # Build models
    features = [
        "s9_difference",
        "h9_difference",
        "ip_ratio",
        "hr9_difference",
    ]
    print("Model from 100-day stats:")
    models.build_models(roll_predictor_list[features], response_series, 1)
    print("Model from prior career stats:")
    models.build_models(career_predictor_list[features], response_series, 2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
