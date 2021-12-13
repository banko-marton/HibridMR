from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np


def _p_score_local(x: pd.Series, y: pd.Series):
    cont_table = pd.crosstab(x, y)
    chi, p, dof, expected = chi2_contingency(cont_table)
    return p


def _p_scores_single_conditional_local(x: pd.Series, y: pd.Series, z: pd.Series):
    z_values = z.unique()
    scores = []
    for z_val in z_values:
        x_local = x[z == z_val]
        y_local = y[z == z_val]
        cont_table = pd.crosstab(x_local, y_local)
        chi, p, dof, expected = chi2_contingency(cont_table)
        scores.append(p)
    return scores


def _p_scores_multi_conditional_local(x: pd.Series, y: pd.Series, z: pd.DataFrame):
    z_values = z.drop_duplicates()
    scores = []
    for _, row_value in z_values.iterrows():
        criteria = np.array(z == row_value)
        criteria = np.all(criteria, axis=1)
        x_local = x[criteria]
        y_local = y[criteria]
        cont_table = pd.crosstab(x_local, y_local)
        chi, p, dof, expected = chi2_contingency(cont_table)
        scores.append(p)
    return scores


def p_chi2_score(data: pd.DataFrame, x_col: str, y_col: str, z_col: (list or tuple or str) = None, method: str = 'min'):
    """
    Calculates the (conditional) p-score between x and y.
    :param data: The data based on which the p-score will be calculated.
    :param x_col: The name of the column representing variable x.
    :param y_col: The name of the column representing variable y.
    :param z_col: Either the name of the column representing variable z, or a list or tuple.
    of names of columns which represent the variable set Z. If given, the p-score will be conditioned by Z.
    :param method: The method based on which the conditional p-score will be calculated. Must be either 'min' or 'avg'.
    :return: The p-score between x and y.
    """
    x, y = data[x_col], data[y_col]
    if z_col is None:
        return _p_score_local(x, y)

    if type(z_col) is str:
        z = data[z_col]
        scores = _p_scores_single_conditional_local(x, y, z)
    elif type(z_col) is list or type(z_col) is tuple:
        z = data[list(z_col)]
        scores = _p_scores_multi_conditional_local(x, y, z)
    else:
        raise TypeError("The Z variable set index must be string, list or tuple.")

    if method == 'min':
        return min(scores)
    if method == 'avg':
        return sum(scores) / len(scores)

    raise AttributeError("The method must be 'min' or 'avg'.")


def is_independent(data: pd.DataFrame, x_col: str, y_col: str, z_col: (list or tuple or str) = None,
                   alpha: float = 0.05, method: str = 'min'):
    """
    Checks whether x and y are independent in the given datased based on the (conditional) p-score between them.
    :param data: The data based on which the p-score will be calculated.
    :param x_col: The name of the column representing variable x.
    :param y_col: The name of the column representing variable y.
    :param z_col: Either the name of the column representing variable z, or a list or tuple.
    :param alpha: The significance level for the p-scores, above which the variables are considered to be independent.
    :param method: The method based on which the conditional p-score will be calculated. Must be either 'min' or 'avg'.
    :return: True if x and y are independent, False if they are dependent.
    """
    return p_chi2_score(data=data, x_col=x_col, y_col=y_col, z_col=z_col, method=method) > alpha
