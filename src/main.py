import logging
import sys
from pathlib import Path
import pandas as pd
import argparse
import marketFeat
import model
import log
import validation
from config import PROXY_TICKERS, hmmParam

parser = argparse.ArgumentParser()


parser.add_argument("-p", "--portfolio",
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
            outPath = log.makeOutPath(args.output, args.portfolio)
        else:
            outPath = Path(args.output)
    except Exception:
        logging.warning("Failed to write final log")
        
    log.writeLogStart(outPath, args.portfolio)
    log.setUpLogging()
    log.configToJson(outPath)
    
    status, url = log.checkConnection(url)
    if status != 0:
        log.writeLogEnd(status=status, url=url)
        
    try: 
        dataFrames = {}
        
        # Get market info
        marketData, featureSets = marketFeat.getMarketFeatures(PROXY_TICKERS)
        portfolio = marketFeat.getStockFeatures(args.portfolio)  
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
                args.portfolio,
                featureName,
                featureCols
            )
            log.saveEnResults(outPath, baselineEnResults, featureName, model="baselineElasticNet")
        
            hmmData, hmmResults, hmmModels = model.runHmm(
                fullData,
                featureName, 
                featureCols,
                hmmParam
            )
            log.saveHmmResults(outPath, hmmData, hmmResults, featureName)
            
            elasticNetHmmProbResults = model.elasticNetHmmProb(
                hmmData,
                args.portfolio,
                featureName,
                featureCols,
                hmmParam["n_components"]
            )
            log.saveEnResults(outPath, elasticNetHmmProbResults, featureName, model="elasticNetHmmProb")
            
            elasticNetRegimeSpecResults = model.elasticNetRegimeSpec(
                hmmData,
                args.portfolio,
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
                "hmmModels": hmmModels
            }
        log.marketsToCsv(outPath, fullData)
        
        # Run validation per market feature per model
        sharpeRecord = []
        
        for featureName, resultSet in modelResults.items():
            validationDict = {}
        
            for modelName, resultsDf in resultSet.items():
                if modelName in ["hmmResults", "hmmModels"]:
                    continue
        
                metrics, sharpeRecord, validationDf, validationDict, logStart = validation.mergeResults(
                    fullData,
                    modelName,
                    featureName,
                    args.portfolio,
                    resultsDf
                )
        
                log.saveValidation(
                    outPath, 
                    modelName, 
                    featureName, 
                    metrics, 
                    args.portfolio)
        
                regimeSharpeDf = validation.calcRegimeSharpes(
                    validationDf,
                    hmmData,
                    periodsPerYear=12
                )
        
                fig = validation.plotRegimeSharpes(
                    regimeSharpeDf,
                    modelName
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
            
            fig = validation.plotEquityCurvesHtml(
                validationDict, 
                hmmData,
                featureName, 
                logStart, 
                title=f"Equity Curve: {featureName}" 
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
        fig = validation.plotSharpes(sharpeRecord)
        if fig is not None:
            log.savePlot(
                outPath,
                fig,
                parent="",
                modelName="",
                featureName="",
                name="sharpeHeatmap.html"
            )
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
