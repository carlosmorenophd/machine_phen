"""Runner to get best parameter with force brute """
import itertools

import pandas as pd


def combination_columns_from_data_frame(df: pd.DataFrame, minimal: int = 1):
    """Combine all columns with min of columns on combination

    Args:
        df (pd.DataFrame): data frame to get the columns
        min (int, optional): minimal of column on group. Defaults to 1.

    Yields:
        _type_: return array of combination
    """
    columns = list(df.columns)
    for r in range(minimal, len(columns) + 1):
        for comb in itertools.combinations(columns, r):
            yield list(comb)
