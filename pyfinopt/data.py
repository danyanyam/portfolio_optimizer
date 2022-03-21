from dataclasses import dataclass, field
from typing import Set, Dict, Iterable
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Stock:
    ticker: str = None
    returns: np.array = None

    def merge(self, rhs):
        assert rhs.ticker == self.ticker
        self.returns = np.vstack(self.returns, rhs.returns)

    def __repr__(self):
        return f'Stock(ticker={self.ticker}, data_len={len(self.returns)})'


@dataclass
class Portfolio:
    stocks: Dict[str, Stock] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    _params: np.array = None

    @property
    def tickers(self) -> Set[str]:
        return tuple(self.stocks.keys())

    @property
    def is_optimized(self) -> bool:
        return len(self.weights) != 0

    @staticmethod
    def from_pandas(stocks, ticker_col: str, returns_col: str):
        _stocks = {}
        stocks_it = stocks.groupby(ticker_col)

        for ticker, stock in stocks_it:

            s = Stock(ticker=ticker, returns=stock.loc[:, returns_col].values.reshape(-1, 1))
            existing_stock = _stocks.get(ticker)

            if not existing_stock:
                _stocks.update({ticker: s})
                continue

            existing_stock.merge(s)
            _stocks.update({ticker: existing_stock})

        return Portfolio(_stocks)

    def get_stocks(self, tickers: Iterable[str]):
        return [self.get_stock(ticker) for ticker in tickers]

    def get_returns(self, tickers: Iterable[str] = None):
        if tickers:
            return np.hstack([stock.returns for stock in self.get_stocks(tickers)])
        return np.hstack([stock.returns for stock in self.stocks.values()])

    def get_stock(self, ticker: str):
        return self.__getitem__(ticker)

    def cov(self, tickers: Iterable[str] = None, annualized=True):
        multiplicator = 253 if annualized else 1
        return np.cov(self.get_returns(tickers), rowvar=0) * multiplicator

    def plot(self):
        _, ax = plt.subplots(figsize=(16, 5), dpi=200)
        ax.plot(
            np.cumprod(1 + self.get_returns(), axis=0),
            alpha=0.3,
            linewidth=2,
            label=self.tickers)

        ax.axhline(1, color='grey', linestyle='dashed')
        ax.grid(alpha=0.2)

        if self.weights is not None:
            ax.plot(
                np.cumprod(1 + self.get_returns() @ self._params, axis=0),
                linewidth=1.5,
                color='black',
                label='weighted')
        ax.set_title('Cumulative return of stocks in Portfolio')
        ax.legend()

    def subset(self, tickers):
        return Portfolio(dict(zip(tickers, self.get_stocks(tickers))))

    def set_weights(self, weights: Dict[str, float]):
        assert set(self.tickers) == set(weights)
        self.weights = weights
        self._params = np.array(list(weights.values()))

    def __iter__(self):
        return iter(self.stocks.items())

    def __getitem__(self, key):
        try:
            assert key in self.tickers
        except AssertionError:
            raise Exception(f'Ticker {key} doesnt exist in portfolio.'
                            f'Possible tickers: {self.tickers}')

        return self.stocks[key]

    def __len__(self):
        return len(self.stocks)

    def __repr__(self) -> str:
        class_name = 'Portfolio'
        description = f'stocks={len(self)}, optimized={self.is_optimized}'

        if len(self) > 10:
            return f'{class_name}({description})'
        return f'{class_name}({description}, tickers={self.tickers})'
