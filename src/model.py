import pandas as pd
import numpy as np
import log
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from config import elasticNetParam, getTimeCv, BACKTEST_WINDOW


def walkForwardIndex(modelData):
    """Yield expanding train windows and one-step test windows for walk-forward modelling."""
    modelData = modelData.sort_values("monthYear").copy()
    for i in range(BACKTEST_WINDOW, len(modelData)):
        trainDf = modelData.iloc[:i].copy()
        testDf = modelData.iloc[i:i + 1].copy()

        if len(trainDf) == 0 or len(testDf) == 0:
            continue

        yield trainDf, testDf


def fitElasticNetCv(X, y):
    """Fit a time-decay weighted Elastic Net CV pipeline."""
    n = len(y)
    
    # Create weighting for time-decay
    weights = np.exp(
        np.linspace(
            elasticNetParam["sample_weight"]["LOWER_WEIGHT"],
            elasticNetParam["sample_weight"]["UPPER_WEIGHT"],
            n
        )
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("enet", ElasticNetCV(
            l1_ratio=elasticNetParam["l1_ratio_grid"],
            cv=getTimeCv(),
            max_iter=elasticNetParam["max_iter"]
        ))
    ])

    model.fit(X, y, enet__sample_weight=weights)
    return model


def baselineEnCv(fullData, stock, featureName, featureCols):
    """
    Train a walk-forward Elastic Net model with time-decay weighting.
    
    This baseline specification uses only the selected feature set and does not
    include HMM-derived regime features.
    """
    logStart = log.manualLogStart(process="baselineEnCv", subprocess=featureName)
    yCol = f"{stock}Return_z"
    
    # Get fwd target to prevent contemporaneous data leakage 
    yColFwd = f"{stock}Return_z_fwd1"
    
    # Extract the columns needed for modelling from fullData
    xCols = featureCols.copy()
    modelCols = ["monthYear"] + xCols + [yCol, yColFwd]
    modelData = fullData[modelCols].copy()
    modelData = modelData.dropna(subset=xCols + [yColFwd])
    
    results = []
    
    # Raise exception if data is of insuffient length for the given window
    if len(modelData) <= BACKTEST_WINDOW:
        raise ValueError("Not enough data for walk-forward Elastic Net fitting.")
    
    # Implement walk-forward Elastic nets with time-decay
    for trainDf, testDf in walkForwardIndex(modelData):
        
        xTrain = trainDf[xCols]
        yTrain = trainDf[yColFwd]

        xTest = testDf[xCols]
        yTest = testDf[yColFwd]

        model = fitElasticNetCv(xTrain, yTrain)

        coef = model.named_steps["enet"].coef_
        coefSeries = pd.Series(coef, index=xTrain.columns)

        pred = model.predict(xTest)[0]

        resultsMetrics = {
            "stock": stock,
            "featureName": featureName,
            "modelName": "baselineElasticNet",
            "monthYear": testDf["monthYear"].iloc[0],
            "actual": yTest.iloc[0],
            "pred": pred,
            "alpha": model.named_steps["enet"].alpha_,
            "l1Ratio": model.named_steps["enet"].l1_ratio_,
        }
        resultsMetrics.update(coefSeries.to_dict())
        results.append(resultsMetrics)
        
    resultsDf = pd.DataFrame(results)
    log.manualLogEnd(logStart, process="baselineEnCv", subprocess=featureName)
    return resultsDf 

def elasticNetHmmProb(fullData, stock, featureName, featureCols, nComponents):
    """
    Train a walk-forward Elastic Net model with time-decay weighting.
    
    This specification augments the selected feature set with lagged HMM regime
    probabilities.
    """
    logStart = log.manualLogStart(process="elasticNetHmmProb", subprocess=featureName)
    
    yCol = f"{stock}Return_z"
    # Get fwd target to prevent contemporaneous data leakage 
    yColFwd = f"{stock}Return_z_fwd1"
    
    # Add regime features from the hmm modelling step
    regimeCols = [f"regimeProb{n}_lag1" for n in range(nComponents - 1)]
    
    # Extract relevant columns from the fullData
    xCols = featureCols.copy() + regimeCols 
    modelCols = ["monthYear"] + xCols + [yCol, yColFwd]
    modelData = fullData[modelCols].copy()
    modelData = modelData.dropna(subset=xCols + [yColFwd])
    
    # Raise exception if data is of insuffient length for the given window
    if len(modelData) <= BACKTEST_WINDOW:
        raise ValueError("Not enough data for walk-forward Elastic Net fitting.")

    results = []
    
    # Implement walk-forward Elastic nets with time-decay
    for trainDf, testDf in walkForwardIndex(modelData):
        xTrain = trainDf[xCols]
        yTrain = trainDf[yColFwd]

        xTest = testDf[xCols]
        yTest = testDf[yColFwd]

        model = fitElasticNetCv(xTrain, yTrain)

        coef = model.named_steps["enet"].coef_
        coefSeries = pd.Series(coef, index=xTrain.columns)

        pred = model.predict(xTest)[0]

        resultsMetrics = {
            "stock": stock,
            "featureName": featureName,
            "modelName": "elasticNetHmmProb",
            "monthYear": testDf["monthYear"].iloc[0],
            "actual": yTest.iloc[0],
            "pred": pred,
            "alpha": model.named_steps["enet"].alpha_,
            "l1Ratio": model.named_steps["enet"].l1_ratio_,
        }
        resultsMetrics.update(coefSeries.to_dict())
        results.append(resultsMetrics)

    resultsDf = pd.DataFrame(results)
    
    log.manualLogEnd(logStart, process="elasticNetHmmProb", subprocess=featureName)
    return resultsDf

def elasticNetRegimeSpec(fullData, stock, featureName, featureCols, nComponents):
    """
    Train a walk-forward Elastic Net model with time-decay weighting.
    
    This specification adds regime-specific interaction terms by scaling each
    feature with lagged HMM regime probabilities.
    """
    logStart = log.manualLogStart(process="elasticNetRegimeSpec", subprocess=featureName)
    yCol = f"{stock}Return_z"
    
    # Get fwd target to prevent contemporaneous data leakage 
    yColFwd = f"{stock}Return_z_fwd1"

    baseCols = featureCols.copy()
    
    # Create regime-specific interaction features using lagged regime probabilities
    regimeCols = [f"regimeProb{n}_lag1" for n in range(nComponents - 1)]
    interactionDfs = []
    interactionMatrix = fullData[baseCols].to_numpy()
    for n in range(nComponents - 1):
        regimeCol = f"regimeProb{n}_lag1"
        regimeVec = fullData[regimeCol].to_numpy()

        regimeAdjMatrix = np.einsum("ij,i->ij", interactionMatrix, regimeVec)

        regimeInteractionDf = pd.DataFrame(
            regimeAdjMatrix,
            index=fullData.index,
            columns=[f"{col}_x_regime{n}" for col in baseCols]
        )
        interactionDfs.append(regimeInteractionDf)
        
    # Add the regime specific columns to the fullData
    interactionDf = pd.concat(interactionDfs, axis=1)
    fullData = pd.concat([fullData, interactionDf], axis=1)

    # Extract columns needed for modelling 
    xCols = baseCols + regimeCols + list(interactionDf.columns)
    modelCols = ["monthYear"] + xCols + [yCol, yColFwd]
    modelData = fullData[modelCols].copy()
    modelData = modelData.dropna(subset=xCols + [yColFwd])

    # Raise exception if data is of insuffient length for the given window
    if len(modelData) <= BACKTEST_WINDOW:
        raise ValueError("Not enough data for walk-forward Elastic Net fitting.")

    results = []

    # Implement walk-forward Elastic nets with time-decay
    for trainDf, testDf in walkForwardIndex(modelData):
        xTrain = trainDf[xCols]
        yTrain = trainDf[yColFwd]

        xTest = testDf[xCols]
        yTest = testDf[yColFwd]

        model = fitElasticNetCv(xTrain, yTrain)

        coef = model.named_steps["enet"].coef_
        coefSeries = pd.Series(coef, index=xTrain.columns)

        pred = model.predict(xTest)[0]

        resultsMetrics = {
            "stock": stock,
            "featureName": featureName,
            "modelName": "elasticNetRegimeSpec",
            "monthYear": testDf["monthYear"].iloc[0],
            "actual": yTest.iloc[0],
            "pred": pred,
            "alpha": model.named_steps["enet"].alpha_,
            "l1Ratio": model.named_steps["enet"].l1_ratio_,
        }
        resultsMetrics.update(coefSeries.to_dict())
        results.append(resultsMetrics)

    resultsDf = pd.DataFrame(results)
    log.manualLogEnd(logStart, process="elasticNetRegimeSpec", subprocess=featureName)
    return resultsDf

def getStateOrder(hmmModel):
    """Order HMM states by volatility to stabilise regime labels across windows."""
    covars = hmmModel.covars_
    covarianceType = hmmModel.covariance_type

    if covarianceType == "diag":
        # shape: (nComponents, nFeatures)
        stateMetric = covars.mean(axis=1)

    elif covarianceType == "full":
        # shape: (nComponents, nFeatures, nFeatures)
        stateMetric = np.array([np.trace(cov) for cov in covars])

    elif covarianceType == "spherical":
        # shape: (nComponents,)
        stateMetric = covars

    elif covarianceType == "tied":
        raise ValueError(
            "Cannot order states by volatility when covariance_type='tied' "
            "because all states share the same covariance structure."
        )

    else:
        raise ValueError(f"Unsupported covariance_type: {covarianceType}")

    stateOrder = np.argsort(stateMetric)
    return stateOrder, stateMetric


def runHmm(fullData, featureName, featureCols, hmmParam):
    """
    Fit a walk-forward Hidden Markov Model to identify latent market regimes.

    The model is trained on the provided feature set and used to infer
    regime probabilities over time.

    Returns both the transformed dataset and fitted HMM models.
    """
    logStart = log.manualLogStart(process="runHmm", subprocess=featureName)
    hmmModels = {}

    hmmCols = ["monthYear"] + featureCols.copy()
    modelData = fullData[hmmCols].dropna().copy()

    if len(modelData) <= BACKTEST_WINDOW:
        raise ValueError(
            "Not enough data to sample, please choose a smaller window range in the config settings."
        )

    nComponents = hmmParam["n_components"]

    # Create output columns in fullData
    fullData = fullData.copy()
    fullData["regime"] = pd.Series(index=fullData.index, dtype="Int64")

    for n in range(nComponents):
        fullData[f"regimeProb{n}"] = pd.Series(index=fullData.index, dtype="float")

    # Walk-forward fitting
    for trainDf, testDf in walkForwardIndex(modelData):
        if len(testDf) != 1:
            raise ValueError("walkForwardIndex must return exactly one test row per step.")

        hmmModel = GaussianHMM(
            n_components=hmmParam["n_components"],
            covariance_type=hmmParam["covariance_type"],
            n_iter=hmmParam["n_iter"],
            random_state=hmmParam["random_state"]
        )

        xTrain = trainDf[featureCols].values
        xTest = testDf[featureCols].values

        hmmModel.fit(xTrain)

        # Raw HMM outputs in arbitrary state order
        testRegimeOld = hmmModel.predict(xTest)[0]
        testProbsOld = hmmModel.predict_proba(xTest)[0]

        # Stabilise state labels by ordering from lowest to highest volatility
        stateOrder, stateMetric = getStateOrder(hmmModel)
        oldToNew = {
            oldState: newState
            for newState, oldState in enumerate(stateOrder)
        }

        testRegime = oldToNew[testRegimeOld]
        testProbs = testProbsOld[stateOrder]

        testIndex = testDf.index[0]

        fullData.loc[testIndex, "regime"] = testRegime

        for n, prob in enumerate(testProbs):
            fullData.loc[testIndex, f"regimeProb{n}"] = prob

        windowKey = f'{trainDf["monthYear"].iloc[0]}_{trainDf["monthYear"].iloc[-1]}'
        hmmModels[windowKey] = {
            "model": hmmModel,
            "trainStart": trainDf["monthYear"].iloc[0],
            "trainEnd": trainDf["monthYear"].iloc[-1],
            "testIndex": testIndex,
            "testRegimeOld": testRegimeOld,
            "testRegime": testRegime,
            "testProbsOld": testProbsOld,
            "testProbs": testProbs,
            "stateOrder": stateOrder,
            "stateMetric": stateMetric,
            "oldToNew": oldToNew
        }

    # Important: sort by monthYear before creating lags
    fullData = fullData.sort_values("monthYear").copy()

    # Create lagged probability columns after all predictions are filled
    for n in range(nComponents):
        colName = f"regimeProb{n}"
        fullData[f"{colName}_lag1"] = fullData[colName].shift(1)

    # Summaries based on rows where regime was assigned
    labelledData = fullData.dropna(subset=["regime"]).copy()
    labelledData["regime"] = labelledData["regime"].astype(int)

    regimeCounts = labelledData["regime"].value_counts().sort_index()
    regimeMean = labelledData.groupby("regime")[featureCols].mean()
    regimeDev = labelledData.groupby("regime")[featureCols].std()

    hmmResults = {
        "featureName": featureName,
        "regimeCounts": regimeCounts,
        "regimeMean": regimeMean,
        "regimeDev": regimeDev
    }
    log.manualLogEnd(logStart, process="runHmm", subprocess=featureName)
    return fullData, hmmResults, hmmModels