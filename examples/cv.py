"""Minimal example of XGBDistribution on California Housing dataset
"""
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from xgboost_distribution import XGBDistribution


def main():
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    model = XGBDistribution(
        distribution="normal",
        natural_gradient=True,
        max_depth=2,
        early_stopping_rounds=10,
    )

    history = model.cv(
        X_train,
        y_train,
        {"eta": 0.1, "subsample": 0.5, "colsample_bytree": 0.5, "seed": 0},
        num_boost_round=500,
        nfold=5,
        verbose_eval=50,
    )
    print(history)


if __name__ == "__main__":
    main()
