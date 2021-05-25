from xgb_dist.distributions.base import BaseDistribution
from xgb_dist.distributions.normal import Normal  # noqa


def get_distributions():
    """Get dict of all available distributions"""
    return {
        subclass.__name__.lower(): subclass
        for subclass in BaseDistribution.__subclasses__()
    }