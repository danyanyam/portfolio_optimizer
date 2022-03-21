from dataclasses import dataclass, field
from typing import List, Set
import numpy as np


@dataclass
class Stock:
    ticker: str = None
    data: np.array = None
    dates: np.array = None


@dataclass
class Portfolio:
    stocks: List[Stock] = field(default_factory=list)

    @staticmethod
    def from_pandas(stocks, date_col: str, ticker_col: str, close_col: str):
        _stocks = []
        stocks_it = stocks.groupby(ticker_col)

        for ticker, stock in stocks_it:

            s = Stock(
                ticker=ticker,
                dates=stock.loc[:, date_col],
                data=stock.loc[:, close_col]
            )

            _stocks.append(s)

        return Portfolio(_stocks)

    def append(self, rhs) -> None:
        self.stocks.extend(rhs.stocks)

    @property
    def tickers(self) -> Set[str]:
        return set(i.ticker for i in self.stocks)

    def __repr__(self) -> str:
        return f'Portfolio(stocks={len(self.stocks)}, tickers={self.tickers})'
