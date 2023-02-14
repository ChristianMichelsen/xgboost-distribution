"""LogNormal distribution
"""
import numpy as np
from scipy.stats import lognorm
from xgboost.compat import PANDAS_INSTALLED, DataFrame

from xgboost_distribution.distributions.base import BaseDistribution
from xgboost_distribution.distributions.utils import check_all_gt_zero


class LogNormal(BaseDistribution):
    """LogNormal distribution with log scoring.

    Definition:

        f(x) = exp( -[ (log(x) - log(scale)) / (2 s^2) ]^2 / 2 ) / s


    with parameters (scale, s).

    We reparameterize:
        s -> log(s) = a
        scale -> log(scale) = b

    Note that b essentially becomes the 'loc' of the distribution:

        log(x/scale) / s = ( log(x) - log(scale) ) / s

    which can then be taken analogous to the normal distribution's

        (x - loc) / scale

    Hence we can re-use the computations in `distribution.normal`, exchanging:

        y -> log(y)
        scale -> s

    """

    @property
    def params(self):
        return ("scale", "s")

    def check_target(self, y):
        check_all_gt_zero(y)

    def gradient_and_hessian(self, y, transformed_params, natural_gradient=True):
        """Gradient and diagonal hessian"""

        log_y = np.log(y)

        loc, log_s = self._split_params(transformed_params)  # note loc = log(scale)
        var = np.exp(2 * log_s)

        grad = np.zeros(shape=(len(y), 2))
        grad[:, 0] = (loc - log_y) / var
        grad[:, 1] = 1 - ((loc - log_y) ** 2) / var

        if natural_gradient:
            fisher_matrix = np.zeros(shape=(len(y), 2, 2))
            fisher_matrix[:, 0, 0] = 1 / var
            fisher_matrix[:, 1, 1] = 2

            grad = np.linalg.solve(fisher_matrix, grad)

            hess = np.ones(shape=(len(y), 2))  # we set the hessian constant
        else:
            hess = np.zeros(shape=(len(y), 2))  # diagonal elements only
            hess[:, 0] = 1 / var
            hess[:, 1] = 2 * ((log_y - loc) ** 2) / var

        return grad, hess

    def loss(self, y, transformed_params):
        scale, s = self.predict(transformed_params)
        return "LogNormal-NLL", -lognorm.logpdf(y, s=s, scale=scale)

    def predict(self, transformed_params):
        log_scale, log_s = self._split_params(transformed_params)
        scale, s = np.exp(log_scale), np.exp(log_s)

        return self.Predictions(scale=scale, s=s)

    def predict_quantiles(
        self,
        transformed_params,
        quantiles=[0.1, 0.5, 0.9],
        string_decimals: int = 2,
        as_pandas=True,
    ):
        if isinstance(quantiles, float):
            quantiles = [quantiles]

        scale, s = self.predict(transformed_params)
        preds = [lognorm(s=s, scale=scale).ppf(q=q) for q in quantiles]

        if as_pandas and PANDAS_INSTALLED:
            index = [f"q_{q:.{string_decimals}f}" for q in quantiles]
            return DataFrame(preds, index=index).T

        else:
            return np.array(preds)

    def starting_params(self, y):
        log_y = np.log(y)
        return np.mean(log_y), np.log(np.std(log_y))

    def _split_params(self, params):
        """Return log_scale (loc) and log_s from params"""
        return params[:, 0], params[:, 1]
