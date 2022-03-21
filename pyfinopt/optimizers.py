import numpy as np
import scipy.optimize as sco

from .data import Portfolio


class BaseOptimizer:
    def __init__(self, portfolio: Portfolio):
        self._init_weights = np.random.random(size=len(portfolio)).reshape(-1, 1)
        self._portfolio = portfolio
        self._normalized = 1 / len(self._init_weights) * np.ones(self._init_weights.shape)
        self._optim_weights = {}
        self.params = None

    def target(self):
        pass

    def constraints(self):
        return [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]

    def bounds(self, short_possible: bool = False):
        w_range = (-1, 1) if short_possible else (0, 1)
        return tuple(w_range for _ in range(len(self._portfolio)))

    def minimize(self, short_allowed: bool = False):
        return sco.minimize(
            fun=self.target,
            x0=self._normalized,
            method='SLSQP',
            bounds=self.bounds(short_allowed),
            constraints=self.constraints()
        )

    def optimize(self, short_allowed: bool = False):
        self.params = None
        results = self.minimize(short_allowed)
        self.params = results['x']
        return dict(zip(self._portfolio.tickers, self.params))


class VolatilityOptimizer(BaseOptimizer):

    def target(self, weights):
        return np.sqrt(weights.T @ self._portfolio.cov(annualized=True)  @ weights)


class SharpeOptimizer(BaseOptimizer):

    def target(self, weights):
        ann_returns = self._portfolio.get_returns().mean(axis=0) * 253
        portfolio_returns = ann_returns @ weights
        portfolio_std = np.sqrt(weights.T @ self._portfolio.cov(annualized=True) @ weights)
        return -1 * (portfolio_returns / portfolio_std)


class EfficientRiskOptimizer(BaseOptimizer):
    pass


class EfficientReturnOptimizer(BaseOptimizer):
    pass


class MaximumDiversificationOptimizer(BaseOptimizer):
    pass
