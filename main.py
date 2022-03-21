import pandas as pd
from pathlib import Path
from pyfinopt.data import Portfolio
from pyfinopt.optimizers import VolatilityOptimizer


def load_data(csv_paths):
    stocks = pd.DataFrame()

    for path in csv_paths:
        _stocks = pd.read_csv(path, parse_dates=True).sort_values('date')
        _stocks['close'] = _stocks['close'].pct_change()
        stocks = pd.concat([stocks, _stocks]).dropna()
    return stocks


def main():

    csv_paths = Path('data').glob('*.csv')
    stocks = load_data(csv_paths)
    portfolio = Portfolio.from_pandas(
        stocks, ticker_col='ticker', returns_col='close')
    print(portfolio)

    opt = VolatilityOptimizer(portfolio)
    opt.optimize()
    portfolio.set_weights(opt.weights)
    print(portfolio.weights)


if __name__ == "__main__":
    main()
