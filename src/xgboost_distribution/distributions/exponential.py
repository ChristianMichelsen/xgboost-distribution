"""Exponential distribution
"""
import numpy as np
from scipy.stats import expon
from xgboost.compat import PANDAS_INSTALLED, DataFrame

from xgboost_distribution.distributions.base import BaseDistribution
from xgboost_distribution.distributions.utils import check_all_ge_zero


class Exponential(BaseDistribution):
    """Exponential distribution with log score

    Definition:

        f(x) = 1 / scale * e^(-x / scale)

    We reparameterize scale -> log(scale) = a to ensure scale >= 0. Gradient:

        d/da -log[f(x)] = d/da -log[1/e^a e^(-x / e^a)]
                        = 1 - x e^-a
                        = 1 - x / scale

    The Fisher information = 1 / scale^2, when reparameterized:

        1 / scale^2 = I ( d/d(scale) log(scale) )^2 = I ( 1/ scale )^2

    Hence we find: I = 1

    """

    @property
    def params(self):
        return ("scale",)

    def check_target(self, y):
        check_all_ge_zero(y)

    def gradient_and_hessian(self, y, transformed_params, natural_gradient=True):
        """Gradient and diagonal hessian"""

        (scale,) = self.predict(transformed_params)

        grad = np.zeros(shape=(len(y), 1))
        grad[:, 0] = 1 - y / scale

        if natural_gradient:
            fisher_matrix = np.ones(shape=(len(y), 1, 1))

            grad = np.linalg.solve(fisher_matrix, grad)
            hess = np.ones(shape=(len(y), 1))  # we set the hessian constant
        else:
            hess = -(grad - 1)

        return grad, hess

    def loss(self, y, transformed_params):
        (scale,) = self.predict(transformed_params)
        return "Exponential-NLL", -expon.logpdf(y, scale=scale)

    def predict(self, transformed_params):
        log_scale = transformed_params  # params are shape (n,)
        scale = np.exp(log_scale)
        return self.Predictions(scale=scale)

    def predict_quantiles(
        self,
        transformed_params,
        quantiles=[0.1, 0.5, 0.9],
        string_decimals: int = 2,
        as_pandas=True,
    ):
        if isinstance(quantiles, float):
            quantiles = [quantiles]

        (scale,) = self.predict(transformed_params)
        preds = [expon(scale=scale).ppf(q=q) for q in quantiles]

        if as_pandas and PANDAS_INSTALLED:
            index = [f"q_{q:.{string_decimals}f}" for q in quantiles]
            return DataFrame(preds, index=index).T

        else:
            return np.array(preds)

    def starting_params(self, y):
        return (np.log(np.mean(y)),)
