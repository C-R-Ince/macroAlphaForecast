import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import log
import networkx as nx
import matplotlib.pyplot as plt
import random
import logging
from config import THRESHOLD, hmmParam, ROLLING_SHARPE_WINDOW

random.seed(42)

# =============================
# HMM based validations 
# =============================

def regimeColourGen():
    """Create per model per regime colour schemata"""
    
    models = [
        "baseline",
        "elasticNetHmmProb",
        "elasticNetRegimeSpec",
        "hiddenMarkovModel"
    ]
    
    n_colours = hmmParam["n_components"]
    regimeColours = {}
    
    for m in models:
        regimeColours[m] = {}
        
        for i in range(n_colours):
            regimeColours[m][f"regime{i}"] = (
                "#" + ''.join(random.choice('0123456789ABCDEF') for _ in range(6))
                )
            
    return regimeColours

def plotTransMat(featureName, meanTransMat, plotTitle):
    """Plot of mean transistion matrices as both heatmaps and nodes per feature"""
    logStart = log.manualLogStart(process=f"Hidden Markov Model based validation", subprocess=featureName)    
    
    transMat = pd.DataFrame(meanTransMat).fillna(0)

    # Heatmap
    fig = px.imshow(
        transMat,
        text_auto=True,
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        labels={
            "x": "Next Regime",
            "y": "Current Regime"
        }
    )

    fig.update_layout(
        title=plotTitle
    )

    return fig, logStart

def plotTransNode(logStart, featureName, meanTransMat, plotTitle):
    """Plot of mean transistion matrices as both heatmaps and nodes per feature"""
    transMat = pd.DataFrame(meanTransMat).fillna(0)

    g = nx.DiGraph()

    for i in range(transMat.shape[0]):
        for j in range(transMat.shape[1]):

            prob = transMat.iloc[i, j]

            if prob > 0.001:
                g.add_edge(i, j, weight=prob)

    fig = plt.figure(figsize=(8, 6))
    
    pos = nx.circular_layout(g)

    edgeLabels = {
        (u, v): f"{d['weight']:.2f}"
        for u, v, d in g.edges(data=True)
    }
    
    edgeWidths = [
        d["weight"] * 5
        for (_, _, d) in g.edges(data=True)
    ]
    
    nx.draw(
        g,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=2500,
        arrows=True,
        width=edgeWidths,
        font_size=12
    )
    
    nx.draw_networkx_edge_labels(
        g,
        pos,
        edge_labels=edgeLabels
    )
    
    plt.title(plotTitle)

    log.manualLogEnd(logStart, process=f"Hidden Markov Model based validation", subprocess=featureName)
    return fig

# =============================
# Data handling for validation 
# =============================

def mergeResults(fullData, modelName, featureName, stock, resultsDf, sharpeRecord, validationDict):
    """Merge model results with forward stock returns and compute validation metrics."""
    logStart = log.manualLogStart(process=f"Model level Elastic Net validation on {modelName}", subprocess=featureName)
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

# ========================================
# Model based Elastic net based validation 
# ========================================

def plotRegimeSharpes(regimeSharpeDf, plotTitle, logStart, modelName, featureName):
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
        title=plotTitle,
        xaxis_title="Regime",
        yaxis_title="Sharpe Ratio",
        barmode="group"
    )
    
    log.manualLogEnd(logStart, process=f"Model level Elastic Net validation on {modelName}", subprocess=featureName)
    return fig

# ===========================================
# Feature based Elastic net based validation 
# ===========================================

def plotEquityCurvesHtml(validationDict, regimeColours, hmmData, plotTitle, featureName):
    """Plot of equity curves by model, per feature"""
    logStart = log.manualLogStart(process=f"Feature level Elastic Net validation", subprocess=featureName)
    
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
        
        regimePeriods = {}
        
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
                    line=dict(
                        dash="dot",
                        color = regimeColours[modelName][f"regime{n}"]
                        )
                )
            )
            
            plotDf["block"] = (
                (plotDf["regime"] != plotDf["regime"].shift())
                .cumsum()
            )
            
            regimePeriods = {}
            
            for block_id, block_df in plotDf.groupby("block"):
            
                regime_value = block_df["regime"].iloc[0]
                if regime_value == n:
                    regimePeriods[block_id] = {
                        f"regime{regime_value}": {
                            "start": block_df["monthYear"].iloc[0],
                            "end": block_df["monthYear"].iloc[-1]
                        }
                    }
                
                    fig.add_vrect(
                        x0=block_df["monthYear"].iloc[0],
                        x1=block_df["monthYear"].iloc[-1],
                        fillcolor=regimeColours[modelName][f"regime{n}"],
                        layer="below",
                        opacity=0.05                
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
        title=plotTitle,
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        hovermode="x unified",
        template="plotly_white"
    )

    return fig, logStart

def plotBacktest(validationDict, regimeColours, hmmData, plotTitle):
    """Plot of backtest results by model per feature"""
    fig = go.Figure()

    for modelName, validationDf in validationDict.items():
        if validationDf.empty:
            continue
        validationDf = validationDf.merge(
            hmmData[["monthYear", "regime"]],
            on="monthYear",
            how="left"
            ).sort_values("monthYear").copy().dropna()
        
        plotDf = validationDf.copy()

        fig.add_trace(
            go.Scatter(
                x=plotDf["monthYear"],
                y=plotDf["actual"],
                mode="lines+markers",
                name=f"{modelName} actual"
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=plotDf["monthYear"],
                y=plotDf["pred"],
                mode="lines+markers",
                line=dict(dash="dash"),
                name=f"{modelName} predicted"
            )
        )
        
        regimePeriods = {}
        
        for n in range(hmmParam["n_components"]):
            plotDf[f"strategyReturnFiltered_regime{n}"] = np.where(
                plotDf["regime"] == n,
                plotDf["strategyReturn"],
                np.nan
            )
            
            plotDf["block"] = (
                (plotDf["regime"] != plotDf["regime"].shift())
                .cumsum()
            )
            
            regimePeriods = {}
            
            for block_id, block_df in plotDf.groupby("block"):
            
                regime_value = block_df["regime"].iloc[0]
                if regime_value == n:
                    regimePeriods[block_id] = {
                        f"regime{regime_value}": {
                            "start": block_df["monthYear"].iloc[0],
                            "end": block_df["monthYear"].iloc[-1]
                        }
                    }
                
                    fig.add_vrect(
                        x0=block_df["monthYear"].iloc[0],
                        x1=block_df["monthYear"].iloc[-1],
                        fillcolor=regimeColours[modelName][f"regime{n}"],
                        layer="below",
                        opacity=0.025                
                        )

    fig.update_layout(
        title=plotTitle,
        xaxis_title="Date",
        yaxis_title="Return",
        hovermode="x unified",
        template="plotly_white"
    )

    return fig

def plotSharpeRolling(validationDict, plotTitle, logStart, featureName):
    """Plot of rolling sharpe line graphs results by model per feature"""
    fig = go.Figure()
    
    commonStart = max(
        df["monthYear"].min()
        for df in validationDict.values()
    )
    
    for modelName, validationDf in validationDict.items():
        if validationDf.empty:
            continue
        
        plotDf = (
            validationDf[
                validationDf["monthYear"] >= commonStart
                ]
            .copy()
        )
        
        plotDf = plotDf[["monthYear", "strategyReturn"]]
    
        plotDf["rollingMean"] = (
            plotDf["strategyReturn"]
            .rolling(ROLLING_SHARPE_WINDOW, min_periods=ROLLING_SHARPE_WINDOW)
            .mean()
        )
        
        plotDf["rollingStd"] = (
            plotDf["strategyReturn"]
            .rolling(ROLLING_SHARPE_WINDOW, min_periods=ROLLING_SHARPE_WINDOW)
            .std()
        )
        
        plotDf["rollingSharpe"] = (
            plotDf["rollingMean"] / plotDf["rollingStd"]
        ) * np.sqrt(12)
        
        
        fig.add_trace(
            go.Scatter(
                x=plotDf["monthYear"],
                y=plotDf["rollingSharpe"],
                mode="lines",
                name=modelName
            )
        )
    
    fig.update_layout(
        title=plotTitle,
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        barmode="group"
    )
    fig.add_hline(y=0, line_dash="dash")
    
    log.manualLogEnd(logStart, process=f"Feature level Elastic Net validation", subprocess=featureName)
    return fig

def plotSharpes(sharpeRecord, plotTitle):
    """Plot a heatmap of Sharpe ratios by model and feature set""" 
    logStart = log.manualLogStart(process=f"Validation", subprocess="Sharpe ratios by model and feature set")   
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
        title=plotTitle
    )
    log.manualLogEnd(logStart, process=f"Validation", subprocess="Sharpe ratios by model and feature set")
    return fig