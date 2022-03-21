# portfolio_optimizer

In this repository, code is written to optimize portfolios on such grounds as:

    1. minimizing variance
    2. maximizing return
    3. maximizing Sharpe ratio
    4. optimizing risk and profitability
    5. maximizing of diversification

Main notebook is [here](main.ipynb). Source code in in folder [pyfinopt](pyfinopt/)
## 🛠 Installation and Dependencies

- `python` 3.9.6
- `pyenv` from [here](https://github.com/pyenv/pyenv)
- `poetry`: ```pip install poetry```
- all the needed packages from `pyproject.toml` and your own `venv`:
    - ```pyenv install 3.9.6 && pyenv local 3.9.6```
    - `poetry` instruction can be found [here](https://blog.jayway.com/2019/12/28/pyenv-poetry-saviours-in-the-python-chaos/)
    - ```poetry update```
- `pyproject.toml` or `requirements.txt` with dependencies needed.