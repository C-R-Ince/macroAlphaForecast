import pandas as pd
import numpy as np
from config import DATA_START_DATE, WINDOW, GPR_LIST
import log
import logging
import yfinance as yf

def getTickerData(ticker):
    """
    Retrieve monthly price data for a ticker.

    Attempts to download data from yfinance first. If unsuccessful,
    falls back to stooq.pl using the original ticker and then a translated ticker.

    Parameters
    ----------
    ticker : str
        Ticker symbol in yfinance format.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing at least 'Date' and 'Close' columns.

    Raises
    ------
    ValueError
        If data cannot be retrieved from any source.
    """
    colsNeeded = ["Date", "Close"]
    df = yf.download(
        ticker,
        start=DATA_START_DATE,
        interval="1mo",
        progress=False,
        auto_adjust=False
        )
    if not df.empty:
        logging.info(f"{ticker} information obtained using yFinance")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        # ensure required columns exist regardless of case differences between sources
        if all(col.lower() in df.columns.str.lower() for col in colsNeeded):
            try: 
                # attempt to retrieve company metadata separately from price data
                stock = yf.Ticker(ticker)
                stockInfo = stock.info
                logging.info(
                    f"Stock information, {ticker}: \n"
                    f"Company: {stockInfo.get('longName', 'NA')}\n"
                    f"Sector: {stockInfo.get('sector', 'NA')}\n"
                    f"Industry: {stockInfo.get('industry', 'NA')}"
                )
            except Exception as e:
                logging.warning(f"Unable to retrieve metadata for {ticker}: {e}") 
            return df
    # If yFinanace unsuccesful, reattempt with stooq
    logging.warning(f"Failed to source {ticker} data from yfinance. Retrying with stooq.")
    stooqUrl = f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=m"
    checkedUrl = log.checkConnection(stooqUrl)
    df = pd.read_csv(checkedUrl)
    # ensure required columns exist regardless of case differences between sources
    if not df.empty and all(col.lower() in df.columns.str.lower() for col in colsNeeded):
        logging.info(f"{ticker} information obtained using stooq.pl")
        return df
    # Attempts to translate the ticker from a yFinance-style to stooq style if unable to return valid data        
    stooqTicker = tickerTranslate(ticker)
    logging.info(f"Retrying {ticker} with translated: {stooqTicker}")
    stooqUrl = f"https://stooq.com/q/d/l/?s={stooqTicker.lower()}&i=m"
    checkedUrl = log.checkConnection(stooqUrl)
    df = pd.read_csv(checkedUrl)
    # ensure required columns exist regardless of case differences between sources
    if not df.empty and all(col.lower() in df.columns.str.lower() for col in colsNeeded):
        logging.info(f"{stooqTicker} information obtained using tickerTranslate() stooq.pl")
        return df
    raise ValueError(f"Unable to obtain information for {ticker}, please check ticker syntax for Yahoo finance or stooq.pl")        

def tickerTranslate(ticker): 
    """
    Convert a yfinance-style ticker to a stooq.pl-compatible ticker.

    Handles crypto, futures, FX, and equity exchange suffix mappings.

    Parameters
    ----------
    ticker : str
        Ticker in yfinance format.

    Returns
    -------
    str
        Translated ticker for stooq.pl.
    """
       
    ticker = ticker.upper()
    
    # NASDAQ
    if ticker == "^IXIC":
        return "^NDQ"
    
    # S&P500
    if ticker == "^GSPC":
        return "^SPX"
    
    # Gold
    if ticker == "GLD":
        return "XAUUSD"
    
    # Silver
    if ticker == "SLV":
        return "XAGUSD"
    
    # crypto
    if "-" in ticker:
        return ticker.replace("-", "")
    
    # futures
    if "=F" in ticker:
        return ticker.replace("=", ".")
    
    # Currency
    if "=X" in ticker:
        return ticker.replace("=X", "")
    
    # equities / ETFs
    if "." in ticker:
        symbol, exchange = ticker.split(".")
        exchangeMap = {
            "L": "UK",
            "DE": "DE",
            "PA": "FR",
            "AS": "NL"
        }
        return f"{symbol}.{exchangeMap.get(exchange, exchange)}"

    # default US
    return f"{ticker}.US"
                
def getMarketFeatures(tickers):
    """Load and engineer market proxy features for model training."""
    logStart = log.manualLogStart(process="Getting market information", subprocess="")
    
    colsReturn = []
    colsZ = []
    colsRoll = []
    colsLag1 = []
    colsLag2 = []
    colsVol = []
    # Source and format GPR data
    url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
    status, url = log.checkConnection(url)
    if status != 0:
        log.writeLogEnd(status=status, url=url)
    gpr = pd.read_excel(url)
    gpr["month"] = pd.to_datetime(gpr["month"])
    gpr = gpr.rename(columns={"month" : "Date"})
    gpr = gpr[gpr["Date"] > pd.Timestamp(DATA_START_DATE)]
    gpr["monthYear"] = gpr["Date"].dt.strftime('%Y-%m')
    # Remove missing values in early historical data
    gpr = gpr.dropna(subset=["GPR"])
    
    # Create GPR features
    for g in GPR_LIST:
        zCol = f"{g}_z"
        rollCol = f"{g}_roll{WINDOW}"
        lag1Col = f"{g}_lag1"
        lag2Col = f"{g}_lag2"       
        
        gpr[g + "_z"] = (gpr[g] - gpr[g].mean()) / gpr[g].std()
        gpr[rollCol] = gpr[zCol].rolling(WINDOW).mean().shift(1)
        gpr[lag1Col] = gpr[zCol].shift(1)
        gpr[lag2Col] = gpr[zCol].shift(2)
        
    gprFeatureCols = []
    for g in GPR_LIST:
        gprFeatureCols.extend([
            f"{g}_z",
            f"{g}_roll{WINDOW}",
            f"{g}_lag1",
            f"{g}_lag2"
        ])
        colsZ.append(f"{g}_z")
        colsRoll.append(f"{g}_roll{WINDOW}")
        colsLag1.append(f"{g}_lag1")
        colsLag2.append(f"{g}_lag2")
        
    # Initialise data frame for combining GPR and proxy features
    marketData = gpr[["Date", "monthYear"] + GPR_LIST + gprFeatureCols]
    
    # Source and format the proxy stocks from config
    for name, ticker in tickers.items():
        df = getTickerData(ticker)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[df["Date"] > pd.Timestamp(DATA_START_DATE)]
        df["monthYear"] = df["Date"].dt.strftime('%Y-%m')
        df = df[["monthYear", "Close"]]
        df = df.rename(columns={
            "Close": f"{name}Close"
        })
        
        # Create proxy stock features
        closeCol = f"{name}Close"
        returnCol = f"{name}Return"
        logCloseCol = f"{name}LogClose"
        logReturnCol = f"{name}LogReturn"
        returnZCol = f"{name}Return_z"
        rollCol = f"{name}_roll{WINDOW}"
        lag1Col = f"{name}_lag1"
        lag2Col = f"{name}_lag2"
        volCol = f"{name}Vol_{WINDOW}"

        df[returnCol] = df[closeCol].pct_change()
        df[logCloseCol] = np.log(df[closeCol])

        # New columns
        df[logReturnCol] = df[logCloseCol].diff()
        df[rollCol] = df[logReturnCol].rolling(WINDOW).mean().shift(1)
        df[lag1Col] = df[logReturnCol].shift(1)
        df[lag2Col] = df[logReturnCol].shift(2)
        df[volCol] = df[logReturnCol].rolling(WINDOW).std().shift(1)
        
        
        colsReturn.append(logReturnCol)
        colsRoll.append(rollCol)
        colsLag1.append(lag1Col)
        colsLag2.append(lag2Col)
        colsVol.append(volCol)
        
        # Merge with dataframe to preserve stock information through loops
        marketData = pd.merge(
            marketData,
            df,
            how="inner",
            on="monthYear"
        )
    
    # Store the features to be used during modelling
    featureSets = {
        "return": colsReturn,
        f"rolling{WINDOW}": colsRoll,
        "lag1": colsLag1,
        "lag2": colsLag2,
        "vol": colsVol
    }
    log.manualLogEnd(logStart, process="Getting market information", subprocess="")
    return marketData, featureSets

def getStockFeatures(stock):
    """Load and engineer features for the target stock."""
    logStart = log.manualLogStart(process="Getting user stock information", subprocess="")
    # Source and format the stock data
    name = stock
    df = getTickerData(name)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"] > pd.Timestamp(DATA_START_DATE)]
    df["monthYear"] = df["Date"].dt.strftime('%Y-%m')
    df = df[["monthYear", "Close"]]
    df = df.rename(columns={
        "Close": f"{name}Close"
    })
    
    # Generate stock features
    closeCol = f"{name}Close"
    returnCol = f"{name}Return"
    logCloseCol = f"{name}LogClose"
    logReturnCol = f"{name}LogReturn"
    returnZCol = f"{name}Return_z"
    returnZColFwd = f"{name}Return_z_fwd1"
    rollCol = f"{name}_roll{WINDOW}"
    lag1Col = f"{name}_lag1"
    lag2Col = f"{name}_lag2"
    
    df[returnCol] = df[closeCol].pct_change()
    df[logCloseCol] = np.log(df[closeCol])
    df[logReturnCol] = df[logCloseCol].diff()
        
    df[returnZCol] = (df[logReturnCol] - df[logReturnCol].mean()) / df[logReturnCol].std()
    df[returnZColFwd] = df[returnZCol].shift(-1)
    df[rollCol] = df[returnZCol].rolling(WINDOW).mean().shift(1)
    df[lag1Col] = df[returnZCol].shift(1)
    df[lag2Col] = df[returnZCol].shift(2)
    
    portfolio = df
    
    log.manualLogEnd(logStart, process="Getting user stock information", subprocess="")
    return portfolio