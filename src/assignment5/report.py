"""
"""
from pathlib import Path


def style_table(styler):
    # Stack Overflow Q#64812819
    # Styler doesn't come with lines, had to add them.
    styler = styler.set_table_styles(
        [
            {"selector": "", "props": [("border", "1px solid grey")]},
            {"selector": "tbody td", "props": [("border", "1px solid grey")]},
            {"selector": "th", "props": [("border", "1px solid grey")]},
        ]
    )
    return styler


def save_table(styler, title, path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
    file_path = path + title
    styler = style_table(styler)
    with open(file_path, "w+") as file:
        file.write(styler.render())
    return None


def add_predictor_links(column_list_metrics):
    p_link = '<a target="_blank" href="'
    p_link += "plots/{0}_feature_plot" + '.html">{0}</a>'
    column_list_metrics["bin_plot"] = column_list_metrics["name"]
    b_link = '<a target="_blank" href="'
    b_link += "plots/{0}_bin_plot" + '.html">LINK</a>'
    styler = column_list_metrics.style.format(
        {
            "name": p_link,
            "bin_plot": b_link,
        },
        escape="html",
        na_rep="NA",
    )
    return styler


def add_corr_links(corrs):
    coef_link = '<a target="_blank" href=plots/corr_plot.html>{0}</a>'
    p_link = '<a target="_blank" href="'
    p_link += "plots/{0}_feature_plot" + '.html">{0}</a>'
    styler = corrs.style.format(
        {
            "x0": p_link,
            "x1": p_link,
            "pearson_coef": coef_link,
        },
        escape="html",
        na_rep="NA",
    )
    return styler


def add_bf_links(bins):
    bf_link = '<a target="_blank" href="'
    bf_link += "plots/{0}_bf_plot" + '.html">{0}</a>'
    p_link = '<a target="_blank" href="'
    p_link += "plots/{0}_feature_plot" + '.html">{0}</a>'
    bins["plot"] = bins["x0"] + "X" + bins["x1"]
    styler = bins.style.format(
        {
            "x0": p_link,
            "x1": p_link,
            "plot": bf_link,
        },
        escape="html",
        na_rep="NA",
    )
    return styler


def report_predictors(column_list_metrics, title, path):
    styler = add_predictor_links(column_list_metrics)
    save_table(styler, title, path)


def report_correlations(corrs, title, path):
    styler = add_corr_links(corrs)
    save_table(styler, title, path)


def report_bruteforces(corrs, title, path):
    styler = add_bf_links(corrs)
    save_table(styler, title, path)
