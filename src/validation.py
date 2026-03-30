import pandas as pd
import numpy as np
from config import THRESHOLD, hmmParam
import plotly.express as px
import plotly.graph_objects as go
import log
import logging

sharpeRecord = []
validationDict = {}

def mergeResults(fullData, modelName, featureName, stock, resultsDf):
    """Merge model results with forward stock returns and compute validation metrics."""
    logStart = log.manualLogStart(process=f"Validation on {modelName}", subprocess=featureName)
    returnCol = f"{stock}Return"
    returnFwdCol = f"{stock}Return_fwd1"

    returnDf = fullData[["monthYear", returnCol]].copy()
    returnDf = returnDf.sort_values("monthYear").copy()
    
    # Shift raw returns so each prediction is evaluated against the next month's realised return
    returnDf[returnFwdCol] = returnDf[returnCol].shift(-1)
    
    validationDf = resultsDf.merge(
        returnDf[["monthYear", returnFwdCol]],
        on="monthYear",
        how="left"
    ).sort_values("monthYear").copy()
    validationDf = validationDf.dropna(subset=[returnFwdCol]).copy()
    
    validationDf["return_fwd1"] = validationDf[returnFwdCol]
    
    # Utilise config.THRESHOLD to create prediction quantiles for long/short/flat signals
    #validationDf["signal"] = np.sign(validationDf["pred"])
    threshold = validationDf["pred"].quantile(THRESHOLD)
    
    validationDf["signal"] = 0
    validationDf.loc[validationDf["pred"] > threshold, "signal"] = 1
    validationDf.loc[validationDf["pred"] < -threshold, "signal"] = -1
    
    validationDf["strategyReturn"] = validationDf["signal"] * validationDf[returnFwdCol]
    validationDf["cumReturn"] = (1 + validationDf["strategyReturn"]).cumprod()
    validationDf["cumBuyHold"] = (1 + validationDf[returnFwdCol]).cumprod()
    
    # Annualised monthly Sharpe ratio
    sharpe = validationDf["strategyReturn"].mean() / validationDf["strategyReturn"].std() * np.sqrt(12)
    
    hitRate = (np.sign(validationDf["pred"]) == np.sign(validationDf["actual"])).mean()
    runningMax = validationDf["cumReturn"].cummax()
    validationDf["drawdown"] = (validationDf["cumReturn"] - runningMax) / runningMax
    maxDrawdown = validationDf["drawdown"].min()

    metrics = {
        "name": modelName,
        "validationDf": validationDf,
        "stock": stock,
        "totalReturn": validationDf["cumReturn"].iloc[-1] - 1,
        "buyHoldReturn": validationDf["cumBuyHold"].iloc[-1] - 1,
        "sharpe": sharpe,
        "hitRate": hitRate,
        "maxDrawdown": maxDrawdown,
        "signal": validationDf["signal"].value_counts(dropna=False)
    }
    
    # Return a sharpe record list for use in downstream plotting
    sharpeRecord.append({
        "modelName": modelName,
        "featureName": featureName,
        "sharpe": sharpe
    })
    
    validationDict[modelName] = validationDf

    return metrics, sharpeRecord, validationDf, validationDict, logStart


def plotSharpes(sharpeRecord):
    """Plot a heatmap of Sharpe ratios by model and feature set"""
    sharpe = pd.DataFrame(sharpeRecord)
    sharpeWide = sharpe.pivot(
        index="modelName",
        columns="featureName",
        values="sharpe"
    )

    sharpeWide = sharpeWide.fillna(0)

    sharpeWide = sharpeWide.loc[
        sharpeWide.mean(axis=1).sort_values(ascending=False).index
    ]

    fig = px.imshow(
        sharpeWide,
        text_auto=True,
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0
    )

    fig.update_layout(
        title="Sharpe Ratio by Model and Feature"
    )
    
    return fig

def calcRegimeSharpes(validationDf, hmmData, periodsPerYear=12):
    """Calculate Sharpe ratios within each regime and across the full sample with returns set to zero outside the active regime."""
    validationDf = validationDf.copy()
        
    validationDf = validationDf.merge(
        hmmData[["monthYear", "regime"]],
        on="monthYear",
        how="left"
        ).sort_values("monthYear").copy()
    
    if "regime" not in validationDf.columns:
        return pd.DataFrame()
    
    validationDf = validationDf.dropna(subset=["regime", "strategyReturn"])
    
    regimeRows = []

    for regimeVal, regimeDf in validationDf.groupby("regime", dropna=True):
        regimeDf = regimeDf.copy()

        subsampleReturns = regimeDf["strategyReturn"].dropna()

        if len(subsampleReturns) < 2 or subsampleReturns.std() == 0:
            subsampleSharpe = np.nan
        else:
            subsampleSharpe = (
                np.sqrt(periodsPerYear)
                * subsampleReturns.mean()
                / subsampleReturns.std()
            )
                
        # Full-sample Sharpe for this regime, with zero returns when the regime is inactive
        activeReturns = np.where(
            validationDf["regime"] == regimeVal,
            validationDf["strategyReturn"],
            0.0
        )
        activeReturns = pd.Series(activeReturns, index=validationDf.index).dropna()

        if len(activeReturns) < 2 or activeReturns.std() == 0:
            activeSharpe = np.nan
        else:
            activeSharpe = (
                np.sqrt(periodsPerYear)
                * activeReturns.mean()
                / activeReturns.std()
            )

        regimeRows.append({
            "regime": regimeVal,
            "nObs": len(regimeDf),
            "subsampleSharpe": subsampleSharpe,
            "activeSharpe": activeSharpe
        })

    regimeSharpeDf = (
        pd.DataFrame(regimeRows)
        .sort_values("regime")
        .reset_index(drop=True)
    )
    return regimeSharpeDf

def plotRegimeSharpes(regimeSharpeDf, modelName):
    """Plot grouped bar charts of subsample and active Sharpe ratios by regime."""
    if regimeSharpeDf.empty:
        logging.error("Unable to perform regime plots, regimeDf empty")
        log.writeLogEnd(status = 6)
        return
    
    fig = go.Figure()

    fig.add_bar(
        x=regimeSharpeDf["regime"].astype(str),
        y=regimeSharpeDf["subsampleSharpe"],
        name="subsampleSharpe"
    )

    fig.add_bar(
        x=regimeSharpeDf["regime"].astype(str),
        y=regimeSharpeDf["activeSharpe"],
        name="activeSharpe"
    )

    fig.update_layout(
        title=f"Sharpe Ratio by Regime: {modelName}",
        xaxis_title="Regime",
        yaxis_title="Sharpe Ratio",
        barmode="group"
    )
    
    return fig

def plotEquityCurvesHtml(validationDict, hmmData, featureName, logStart, title, goodRegime=2):
    fig = go.Figure()

    buyHoldAdded = False

    for modelName, validationDf in validationDict.items():
        if validationDf.empty:
            continue
        validationDf = validationDf.merge(
            hmmData[["monthYear", "regime"]],
            on="monthYear",
            how="left"
            ).sort_values("monthYear").copy().dropna()
        
        plotDf = validationDf.copy()

        if "strategyReturn" not in plotDf.columns:
            raise ValueError(f"'strategyReturn' missing for model {modelName}")

        plotDf["cumReturn"] = (1 + plotDf["strategyReturn"].fillna(0)).cumprod()

        fig.add_trace(
            go.Scatter(
                x=plotDf["monthYear"],
                y=plotDf["cumReturn"],
                mode="lines",
                name=modelName
            )
        )
        
        for n in range(hmmParam["n_components"]):
            plotDf[f"strategyReturnFiltered_regime{n}"] = np.where(
                plotDf["regime"] == n,
                plotDf["strategyReturn"],
                np.nan
            )
        
            plotDf[f"cumReturnFiltered_regime{n}"] = (
                1 + plotDf[f"strategyReturnFiltered_regime{n}"]
            ).cumprod()
        
            fig.add_trace(
                go.Scatter(
                    x=plotDf["monthYear"],
                    y=plotDf[f"cumReturnFiltered_regime{n}"],
                    mode="lines",
                    name=f"{modelName}_regime{n}",
                    line=dict(dash="dot")
                )
            )

        if not buyHoldAdded:
            buyHoldCol = None

            if "return_fwd1" in plotDf.columns:
                buyHoldCol = "return_fwd1"
            elif "actual" in plotDf.columns:
                buyHoldCol = "actual"

            if buyHoldCol is not None:
                plotDf["cumBuyHold"] = (1 + plotDf[buyHoldCol]).cumprod()

                fig.add_trace(
                    go.Scatter(
                        x=plotDf["monthYear"],
                        y=plotDf["cumBuyHold"],
                        mode="lines",
                        name="buyHold"
                    )
                )
                buyHoldAdded = True

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        hovermode="x unified",
        template="plotly_white"
    )

    log.manualLogEnd(logStart, process=f"Validation on {modelName}", subprocess=featureName)
    return fig
