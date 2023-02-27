"""Poisson distribution
"""
import numpy as np
from scipy.stats import poisson
from xgboost.compat import PANDAS_INSTALLED, DataFrame

from xgboost_distribution.distributions import base, utils


class Poisson(base.BaseDistribution):
    """Poisson distribution with log score

    Definition:

        f(k) = e^(-mu) mu^k / k!

    We reparameterize mu -> log(mu) = a to ensure mu >= 0. Gradient:

        d/da -log[f(k)] = e^a - k  = mu - k

    The Fisher information = 1 / mu, which needs to be expressed in the
    reparameterized form:

        1 / mu = I ( d/dmu log(mu) )^2 = I ( 1/ mu )^2

    Hence we find: I = mu

    """

    @property
    def params(self):
        return ("mu",)

    def check_target(self, y):
        utils.check_all_integer(y)
        utils.check_all_ge_zero(y)

    def gradient_and_hessian(
        self,
        y,
        transformed_params,
        natural_gradient=True,
        gradient_method="None",
    ):
        """Gradient and diagonal hessian"""

        (mu,) = self.predict(transformed_params)

        grad = np.zeros(shape=(len(y), 1))
        grad[:, 0] = mu - y

        if natural_gradient:
            fisher_matrix = np.zeros(shape=(len(y), 1, 1))
            fisher_matrix[:, 0, 0] = mu

            grad = np.linalg.solve(fisher_matrix, grad)
            grad = utils.stabilize_derivative(gradient=grad, method=gradient_method)

            hess = np.ones(shape=(len(y), 1))  # we set the hessian constant
        else:
            hess = mu

        return grad, hess

    def loss(self, y, transformed_params):
        (mu,) = self.predict(transformed_params)
        return "Poisson-NLL", -poisson.logpmf(y, mu=mu)

    def predict(self, transformed_params):
        log_mu = transformed_params  # params are shape (n,)
        mu = np.exp(log_mu)
        return self.Predictions(mu=mu)

    def predict_quantiles(
        self,
        transformed_params,
        quantiles=[0.1, 0.5, 0.9],
        string_decimals: int = 2,
        as_pandas=True,
    ):
        if isinstance(quantiles, float):
            quantiles = [quantiles]

        (mu,) = self.predict(transformed_params).mu
        preds = [poisson(mu=mu).ppf(q=q) for q in quantiles]

        if as_pandas and PANDAS_INSTALLED:
            index = [f"q_{q:.{string_decimals}f}" for q in quantiles]
            return DataFrame(preds, index=index).T

        else:
            return np.array(preds)

    def starting_params(self, y):
        return (np.log(np.mean(y)),)
