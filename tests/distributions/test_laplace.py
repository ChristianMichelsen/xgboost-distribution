import pytest

import numpy as np

from xgboost_distribution.distributions import Laplace


@pytest.fixture
def laplace():
    return Laplace()


@pytest.mark.parametrize(
    "y, transformed_params, natural_gradient, expected_grad",
    [
        (
            np.array([0, 0]),
            np.array([[0, 1], [1, 0]]),
            True,
            np.array([[0, 1], [1, 0]]),
        ),
        (
            np.array([0, 0]),
            np.array([[0, 1], [1, 0]]),
            False,
            np.array([[0, 1], [1, 0]]),
        ),
    ],
)
def test_gradient_calculation(
    laplace,
    y,
    transformed_params,
    natural_gradient,
    expected_grad,
):
    grad, hess = laplace.gradient_and_hessian(
        y,
        transformed_params,
        natural_gradient=natural_gradient,
    )
    np.testing.assert_array_equal(grad, expected_grad)


def test_loss(laplace):
    loss_name, loss_values = laplace.loss(
        # fmt: off
        y=np.array([1, ]),
        transformed_params=np.array([[1, np.log(1)], ]),
    )
    assert loss_name == "Laplace-NLL"
    np.testing.assert_approx_equal(loss_values, np.array([-np.log(0.5)]))
