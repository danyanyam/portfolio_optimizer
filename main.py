from pathlib import Path
from pyfinopt.data import Portfolio
import pandas as pd


def main():

    csv_paths = Path('data').glob('*.csv')
    portfolio = Portfolio()

    for path in csv_paths:
        p = Portfolio.from_pandas(stocks=pd.read_csv(path, parse_dates=True),
                                  date_col='date', ticker_col='ticker',
                                  close_col='close')
        portfolio.append(p)

    print(portfolio)


if __name__ == "__main__":
    main()
