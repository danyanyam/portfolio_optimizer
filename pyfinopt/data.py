import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Set, Dict, Iterable, List, Optional


@dataclass
class Stock:
    """ Object, that represents stock data model """

    ticker: str = None
    returns: np.array = None

    def merge(self, rhs) -> None:
        """ Merges two stocks in one, by concatting their returns

        Arguments:
            rhs :: Stock,
                another stock to merge
        """

        assert rhs.ticker == self.ticker
        self.returns = np.vstack(self.returns, rhs.returns)

    def __len__(self):
        return len(self.returns)

    def __repr__(self):
        return f'Stock(ticker={self.ticker}, data_len={len(self.returns)})'


@dataclass
class Portfolio:
    """Object that represents collection of stocks. The critical
    assumption of the portfolio is that


    Returns:
        _type_: _description_
    """
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
        """Creates portfolio object from pandas dataframe. Critical assumption:
        all of the stocks have the same date range and same length.

        Arguments:
            stocks :: pd.DataFrame
                data with stocks
            ticker_col :: str
                column name of column with ticker names
            returns_col :: str
                columns name of column with returns

        Returns:
            Portfolio
        """
        _stocks = {}
        stocks_it = stocks.dropna().groupby(ticker_col)

        for ticker, stock in stocks_it:

            s = Stock(ticker=ticker, returns=stock.loc[:, returns_col].values.reshape(-1, 1))
            existing_stock = _stocks.get(ticker)

            if not existing_stock:
                _stocks.update({ticker: s})
                continue

            existing_stock.merge(s)
            _stocks.update({ticker: existing_stock})

        lengths = set(len(stock) for stock in _stocks.values())

        if len(lengths) != 1:
            raise Exception('Cant create portfolio from '
                            f'different returns sizes: {lengths}')

        return Portfolio(_stocks)

    def add_stock(self, stock: Stock) -> None:
        """Adds stock to the portfolio

        Args:
            stock :: Stock
                stock to add to the portfolio

        Raises:
            Exception:
                in case stock is not np.array of size of other stocks

        """
        assert isinstance(stock.returns, np.array), 'Only np arrays are allowed'
        assert len(stock.returns.shape) == 2, 'Only 2-dimensional return vectors allowed'
        assert stock.returns.shape[-1] == 1, 'Only 2-dimensional vector-columns are allowed'

        if len(self.tickers) == 0:
            self.stocks[stock.ticker] = stock
            return

        some_stock = self.stocks[self.tickers[0]]
        if len(stock) != len(some_stock):
            raise Exception('Cant add stock due to the size mismatch '
                            f'allowed returns shape is {len(some_stock)} '
                            f'but received: {len(stock)}')

        existing_stock = self.stocks.get(stock.ticker)

        if existing_stock != None:
            existing_stock.merge(stock)
            stock = existing_stock

        self.stocks[stock.ticker] = stock

    def get_stocks(self, tickers: Iterable[str]) -> List[Stock]:
        """Returns list of Stock objects within the portfolio

        Args:
            tickers :: Iterable[str],
                Iterable with tickers to select stocks by

        Returns:
            List[Stock],
                List of selected stocks
        """
        return [self.get_stock(ticker) for ticker in tickers]

    def get_returns(self, tickers: Optional[Iterable[str]] = None) -> np.array:
        """Returns matrix of returns for all stocks

        Args:
            tickers :: Iterable[str], optional
                list of tickers to select stocks from. If empty, then all returns are returned

        Returns:
            np.array,
                returns of all stocks

        """
        if tickers:
            return np.hstack([stock.returns for stock in self.get_stocks(tickers)])
        return np.hstack([stock.returns for stock in self.stocks.values()])

    def get_stock(self, ticker: str) -> Stock:
        """Returns stock from portfolio by ticker

        Args:
            ticker :: str,
                ticker of stock in the portfolio to return

        Returns:
            Stock
        """
        return self.__getitem__(ticker)

    def cov(self, tickers: Optional[Iterable[str]] = None, annualized: bool = True) -> np.array:
        """Calculates the annualized covariance of columns of returns of
        specified tickers. If no tickers specified, then covariance of all returns
        in portfolio are calculated.

        Args:
            tickers :: Iterable[str], optional
                list of tickers to select stocks from. If empty, then all returns are returned
            annualized :: bool, optional
                whether to calculate annualized returns or not

        Returns:
            np.array: covariance matrix
        """
        multiplicator = 253 if annualized else 1
        return np.cov(self.get_returns(tickers), rowvar=0) * multiplicator

    def plot(self) -> None:
        """Plots cumulated return of stocks in the portfolio.
        If portfolio have been optimized, then portfolio performance
        is also shown.
        """
        _, ax = plt.subplots(figsize=(16, 5), dpi=200)
        ax.plot(
            np.cumprod(1 + self.get_returns(), axis=0),
            alpha=0.3,
            linewidth=2,
            label=self.tickers)

        ax.axhline(1, color='grey', linestyle='dashed')
        ax.grid(alpha=0.2)

        if self.is_optimized:
            ax.plot(
                np.cumprod(1 + self.get_returns() @ self._params, axis=0),
                linewidth=1.5,
                color='black',
                label='weighted')
        ax.set_title('Cumulative return of stocks in Portfolio')
        ax.legend()

    def subset(self, tickers: Iterable[str]):
        """Copies portfolio with specified tickers """
        return Portfolio(dict(zip(tickers, self.get_stocks(tickers))))

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Sets weights of stocks in the portfolio if weights correspond
        to the stocks inside portfolio

        Args:
            weights :: Dict[str, float]
                weights of stocks inside the portfolio
        """
        assert set(self.tickers) == set(weights), 'Sets of tickers dont correspond'
        self.weights = weights
        self._params = np.array(list(weights.values()))

    def __iter__(self):
        """ to iterate over portfolio """
        return iter(self.stocks.items())

    def __getitem__(self, key: str) -> Stock:
        try:
            assert key in self.tickers
        except AssertionError:
            raise Exception(f'Ticker {key} doesnt exist in portfolio.'
                            f'Possible tickers: {self.tickers}')

        return self.stocks[key]

    def __len__(self) -> int:
        return len(self.stocks)

    def __repr__(self) -> str:
        class_name = 'Portfolio'
        description = f'stocks={len(self)}, optimized={self.is_optimized}'

        if len(self) > 10:
            return f'{class_name}({description})'
        return f'{class_name}({description}, tickers={self.tickers})'
