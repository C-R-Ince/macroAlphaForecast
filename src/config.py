from sklearn.model_selection import TimeSeriesSplit

# Date and window settings
DATA_START_DATE = "1999-01-01" # If stocks used are more recent, the earliest date that is shared across all will be used
WINDOW = 6 # Rolling window size in months for market features and portfolio calculations
BACKTEST_WINDOW = 60 # Backtest training window in months. Recommended: 60
THRESHOLD=0.3 # Prediction quantile threshold used to generate trading signals

# Market proxy tickers
PROXY_TICKERS = {
    "gold": "GLD",
    "silver": "SLV",
    "sp500": "^GSPC",
    "tech": "QQQ",
    "energy": "XLE",
    "usd": "DX-Y.NYB",
    "us10y": "^TNX"
}

# Geopolitical Risk (GPR) series from Caldara and Iacoviello (2022).
GPR_LIST = ["GPR", "GPRC_USA"]

# Elastic Net settings
elasticNetParam = {
  "type": "ElasticNetCV",
  "sample_weight": {
    "UPPER_WEIGHT": 0,
    "LOWER_WEIGHT": -4
  },
  "l1_ratio_grid": [0.05, 0.1, 0.3, 0.5, 0.7, 0.9],
  "TimeSeriesSplit": {
    "n_splits": 5,
    "gap": 6
  },
  "max_iter": 1000000
}
def getTimeCv():
    return TimeSeriesSplit(n_splits=elasticNetParam["TimeSeriesSplit"]["n_splits"], gap=elasticNetParam["TimeSeriesSplit"]["gap"])

# Hidden Markov Model settings
hmmParam = {
  "n_components": 2,
  "covariance_type": "full", 
  "n_iter": 10000,
  "random_state": 42
}
