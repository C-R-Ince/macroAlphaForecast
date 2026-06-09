import logging
import sys
from pathlib import Path
import pandas as pd
import argparse
import marketFeat
import model
import log
import validation
from config import PROXY_TICKERS, hmmParam, ROLLING_SHARPE_WINDOW
#from distutils.tests import here

parser = argparse.ArgumentParser()


parser.add_argument("-t", "--ticker",
                    required = True,
                    help = "String with yFinance stock code")

parser.add_argument("-o", "--output",
                    help = "Output path for out directory, defaults to ./audit/")

def main():
    """Run the full modelling, validation, and audit pipeline."""
    status = 0
    url = None
    args = parser.parse_args()
    # Create daemon background thread to update RAM usage
    log.startRamMonitor(interval=5)
    
    # Build output path from command-line input or use the default
    try:
        if args.output is None:
            outPath = log.makeOutPath(args.output, args.ticker)
        else:
            outPath = Path(args.output)
    except Exception:
        logging.warning("Failed to write final log")
        
    log.writeLogStart(outPath, args.ticker)
    log.setUpLogging()
    log.configToJson(outPath)
    
    status, url = log.checkConnection(url)
    if status != 0:
        log.writeLogEnd(status=status, url=url)
        
    try: 
        dataFrames = {}
        
        # Get market info
        marketData, featureSets = marketFeat.getMarketFeatures(PROXY_TICKERS)
        portfolio = marketFeat.getStockFeatures(args.ticker)  
        # Merge all market information and save for audit
        fullData = pd.merge(
            portfolio,
            marketData,
            how="inner",
            on="monthYear"
        )
        
        dataFrames = {
            "marketData": marketData,
            "targetData": portfolio
        }
        log.marketsToCsv(outPath, dataFrames)

        # Modelling per market feature
        modelResults = {}
        for featureName, featureCols in featureSets.items():
            baselineEnResults = model.baselineEnCv(
                fullData,
                args.ticker,
                featureName,
                featureCols
            )
            log.saveEnResults(outPath, baselineEnResults, featureName, model="baselineElasticNet")
        
            hmmData, hmmResults, hmmModels, meanTransMat = model.runHmm(
                fullData,
                featureName, 
                featureCols,
                hmmParam
            )
            log.saveHmmResults(outPath, hmmData, hmmResults, meanTransMat, featureName)
            
            elasticNetHmmProbResults = model.elasticNetHmmProb(
                hmmData,
                args.ticker,
                featureName,
                featureCols,
                hmmParam["n_components"]
            )
            log.saveEnResults(outPath, elasticNetHmmProbResults, featureName, model="elasticNetHmmProb")
            
            elasticNetRegimeSpecResults = model.elasticNetRegimeSpec(
                hmmData,
                args.ticker,
                featureName,
                featureCols,
                hmmParam["n_components"]
            )
            log.saveEnResults(outPath, elasticNetRegimeSpecResults, featureName, model="elasticNetRegimeSpec")
        
            modelResults[featureName] = {
                "baseline": baselineEnResults,
                "elasticNetHmmProb": elasticNetHmmProbResults,
                "elasticNetRegimeSpec": elasticNetRegimeSpecResults,
                "hmmResults": hmmResults,
                "hmmModels": hmmModels,
                "hmmTransMat": meanTransMat
            }
        log.marketsToCsv(outPath, fullData)
        
        # Run validation per market feature per model
        sharpeRecord = []
        regimeColours = validation.regimeColourGen()
        
        for featureName, resultSet in modelResults.items():
            validationDict = {}
        
            for modelName, resultsDf in resultSet.items():
                
                if modelName in ["hmmResults", "hmmModels"]:
                    continue
                
                if modelName == "hmmTransMat":
                    plotTitle=f"Transition Matrix heatmap of {featureName}"
                    fig, logStart = validation.plotTransMat(
                        featureName,
                        meanTransMat=resultsDf, 
                        plotTitle=plotTitle
                        )
                    if fig is not None:
                        log.savePlot(
                            outPath,
                            fig,
                            parent="model",
                            modelName="hiddenMarkovmodel",
                            featureName=featureName,
                            name="meanTransitionMatrix.html"
                        )
                    else:
                        logging.error(f"{plotTitle} was empty, skipping writing step.")
                        
                    plotTitle=f"Transition Nodes of {featureName}"    
                    fig = validation.plotTransNode(
                        logStart,
                        featureName,
                        meanTransMat=resultsDf, 
                        plotTitle=plotTitle
                        )       
                    if fig is not None:
                        log.savePng(
                            outPath,
                            fig,
                            parent="model",
                            modelName="hiddenMarkovmodel",
                            featureName=featureName,
                            name="meanTransitionGraph.png"
                        )
                    else:
                        logging.error(f"{plotTitle} was empty, skipping writing step.")
                    continue 
                
                metrics, sharpeRecord, validationDf, validationDict, logStart = validation.mergeResults(
                    fullData,
                    modelName,
                    featureName,
                    args.ticker,
                    resultsDf,
                    sharpeRecord,
                    validationDict
                )
        
                log.saveValidation(
                    outPath, 
                    modelName, 
                    featureName, 
                    metrics, 
                    args.ticker)
        
                regimeSharpeDf = validation.calcRegimeSharpes(
                    validationDf,
                    hmmData,
                    periodsPerYear=12
                )
        
                plotTitle=f"Sharpe Ratio by Regime: {modelName}"
                fig = validation.plotRegimeSharpes(
                    regimeSharpeDf,
                    plotTitle,
                    logStart,
                    modelName, 
                    featureName
                )        
                if fig is not None:
                    log.savePlot(
                        outPath,
                        fig,
                        parent="model",
                        modelName=modelName,
                        featureName=featureName,
                        name="regimeSharpe.html"
                    )
                else: 
                    logging.error(f"{plotTitle} was empty, skipping writing step.")
            
            plotTitle=f"Equity Curve: {featureName}" 
            fig, logStart = validation.plotEquityCurvesHtml(
                validationDict,
                regimeColours, 
                hmmData,
                plotTitle=plotTitle,
                featureName=featureName
            )
            if fig is not None:
                log.savePlot(
                    outPath,
                    fig,
                    parent="feature",
                    modelName="",
                    featureName=featureName,
                    name="equityCurves.html"
                )
            else:
                logging.error(f"{plotTitle} was empty, skipping writing step.")
            
            plotTitle=f"Backtest: actual vs predicted ({featureName})" 
            fig = validation.plotBacktest(
                validationDict,
                regimeColours, 
                hmmData,
                plotTitle=plotTitle
            )
            if fig is not None:
                log.savePlot(
                    outPath,
                    fig,
                    parent="feature",
                    modelName="",
                    featureName=featureName,
                    name="backtest.html"
                )
            else:
                logging.error(f"{plotTitle} was empty, skipping writing step.")
                           
            plotTitle=f"Rolling sharpe per {ROLLING_SHARPE_WINDOW} months ({featureName})"
            fig = validation.plotSharpeRolling(
                validationDict,
                plotTitle=plotTitle,
                logStart=logStart,
                featureName=featureName
            )        
            if fig is not None:
                log.savePlot(
                    outPath,
                    fig,
                    parent="feature",
                    modelName="",
                    featureName=featureName,
                    name="rollingSharpe.html"
                )
            else:
                logging.error(f"{plotTitle} was empty, skipping writing step.")
                
                
        plotTitle="Sharpe Ratio by Model and Feature"            
        fig = validation.plotSharpes(
            sharpeRecord,
            plotTitle=plotTitle
            )
        if fig is not None:
            log.savePlot(
                outPath,
                fig,
                parent="",
                modelName="",
                featureName="",
                name="sharpeHeatmap.html"
            )
        else:
            logging.error(f"{plotTitle} was empty, skipping writing step.")
                            
        log.writeLogEnd(status)
        
        return status
    
    except Exception:
        status = 1
        logging.exception("Pipeline failed")
        log.writeLogEnd(status)
        raise

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        logging.exception("Pipeline failed")
        sys.exit(1)
