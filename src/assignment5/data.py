"""
"""
import warnings

import pandas as pd
import pymysql

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def load_responses() -> pd.Series:
    query = """
    SELECT
        game_id,
        winner_home_or_away AS home_team_wins
    FROM
        boxscore
    """
    sql_db = pymysql.connect(host="localhost", user=" ", password=" ", db="baseball")

    df = pd.read_sql_query(query, sql_db)
    df = df.replace("H", 1)
    df = df.replace("A", 0)
    df = df.replace("", float("NaN"))
    df = df.dropna()
    df = df.set_index("game_id")
    df = df.sort_index()
    df = df.astype("int")
    series = df["home_team_wins"]

    sql_db.close()
    return series


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(
        labels=[
            "local_date",
            "home_pitcher",
            "away_pitcher",
        ],
        axis=1,
    )
    return df


def load_rolling_100_pitcher_columns() -> pd.DataFrame:
    query = "SELECT * FROM starting_pitcher_stats_100"
    sql_db = pymysql.connect(host="localhost", user=" ", password=" ", db="baseball")

    df = pd.read_sql_query(query, sql_db)
    df = drop_columns(df)
    df = df.set_index("game_id")

    sql_db.close()
    return df


def load_prior_career_pitcher_columns() -> pd.DataFrame:
    query = "SELECT * FROM starting_pitcher_stats_prior_career"

    sql_db = pymysql.connect(host="localhost", user=" ", password=" ", db="baseball")

    df = pd.read_sql_query(query, sql_db)
    df = drop_columns(df)
    df = df.set_index("game_id")

    sql_db.close()
    return df


if __name__ == "__main__":
    y = load_responses()
    X0 = load_rolling_100_pitcher_columns()
    X1 = load_prior_career_pitcher_columns()
    print(f"{y}\n{X0}\n{X1}")
