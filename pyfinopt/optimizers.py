import numpy as np
import scipy.optimize as sco
from typing import Tuple
from abc import ABCMeta, abstractmethod

from .data import Portfolio


class BaseOptimizer(metaclass=ABCMeta):
    """ Base class for optimizers """

    def __init__(self, portfolio: Portfolio):
        self._init_weights = np.random.random(size=len(portfolio)).reshape(-1, 1)  # k x 1
        self._portfolio = portfolio
        self._normalized = 1 / len(self._init_weights) * np.ones(self._init_weights.shape)
        self.weights = {}
        self.constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
        self.params = None
        self._optim = None

    @abstractmethod
    def target(self, weights: np.array) -> float:
        """ Must define the function to be optimized """

    def _bounds(self, short_possible: bool = False) -> Tuple:
        """ Bounds on weights """
        w_range = (-1, 1) if short_possible else (0, 1)
        return tuple(w_range for _ in range(len(self._portfolio)))

    def _minimize(self, short_allowed: bool = False):
        return sco.minimize(
            fun=self.target,
            x0=self._normalized,
            method='SLSQP',
            bounds=self._bounds(short_allowed),
            constraints=self.constraints
        )

    def weighted_return(self, weights: np.array) -> np.array:
        return (self._portfolio.get_returns().mean(axis=0) * 253) @ weights

    def weighted_covariance_return(self, weights: np.array) -> np.array:
        return weights.T @ self._portfolio.cov(annualized=True)  @ weights

    def optimize(self, short_allowed: bool = False):
        self._params = None
        self._optim = self._minimize(short_allowed)
        self._params = self._optim['x']
        self.weights = dict(zip(self._portfolio.tickers, self._params))

    def __repr__(self):
        cls_name = self.__class__.__name__

        if self._optim:
            return f'{cls_name}(optimized={True}, value={self._optim["fun"]})'
        return f'{cls_name}(optimized={False})'


class VolatilityOptimizer(BaseOptimizer):
    """ minimizes volatility """

    def target(self, weights: np.array) -> float:
        return np.sqrt(self.weighted_covariance_return(weights))


class ReturnOptimizer(BaseOptimizer):
    """ maximizes return """

    def target(self, weights: np.array) -> float:
        return self.weighted_return(weights)


class SharpeOptimizer(BaseOptimizer):
    """ maximizes sharpe ratio """

    def target(self, weights: np.array) -> float:
        return_ = self.weighted_return(weights)
        std = np.sqrt(self.weighted_covariance_return(weights))
        return -1 * (return_ / std)


class EfficientRiskOptimizer(VolatilityOptimizer):
    """ minimizes risk given target return """

    def __init__(self, target_return: float, portfolio: Portfolio):
        super().__init__(portfolio)
        self.constraints.append(
            {'type': 'eq', 'fun': lambda w: self.weighted_return(w) - target_return}
        )


class EfficientReturnOptimizer(ReturnOptimizer):
    """ maximizes return given target risk """

    def __init__(self, target_risk: float, portfolio: Portfolio):
        super().__init__(portfolio)
        self.constraints.append(
            {'type': 'eq',
             'fun': lambda w: np.sqrt(self.weighted_covariance_return(w)) - target_risk}
        )


class MaximumDiversificationOptimizer(BaseOptimizer):
    """ maximizes diversification """

    def target(self, weights: np.array) -> float:
        weighted_volatilities = weights.T @ self._portfolio.get_returns().std(axis=0)
        std = np.sqrt(self.weighted_covariance_return(weights))
        return -1 * (weighted_volatilities / std)
