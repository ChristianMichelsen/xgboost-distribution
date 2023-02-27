import numpy as np

"""Utility functions for distributions
"""


def check_all_integer(x):
    if not all(x == x.astype(int)):
        raise ValueError("All values of target must be integers!")


def check_all_ge_zero(x):
    if not all(x >= 0):
        raise ValueError("All values of target must be >=0!")


def check_all_gt_zero(x):
    if not all(x > 0):
        raise ValueError("All values of target must be > 0!")


def stabilize_derivative(gradient: np.ndarray, method: str = "None"):
    """Function that stabilizes gradients.
    ----------
    input_der : np.ndarray
        Gradient.
    method: str
        Stabilization method. Can be either "None", "L1" or "L2".
    Returns
    -------
    gradient_out : np.ndarray
        Stabilized gradient.
    """

    if method == "L1":
        div = np.nanmedian(np.abs(gradient - np.nanmedian(gradient)))
        div = np.where(div < 1e-04, 1e-04, div)
        stab_der = gradient / div

    elif method == "L2":
        div = np.sqrt(np.nanmean(gradient**2))
        div = np.where(div < 1e-04, 1e-04, div)
        div = np.where(div > 10000, 10000, div)
        stab_der = gradient / div

    else:
        stab_der = gradient

    return stab_der