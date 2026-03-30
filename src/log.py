from datetime import datetime
from pathlib import Path
import requests
import threading
import time
import psutil
import os
import config
import json
import logging
import warnings

startTime = time.time()
startDateTime = time.strftime("%Y-%m-%d %H:%M:%S")
peakRamMb = 0

statusMapper = {
    0: "Pipeline Successfully Run",
    1: "Pipeline Failed",
    2: "Can't connect to {url}, please check your connection and try again",
    3: "Requests to {url} timed out, please check your connection and try again",
    4: "Electric Net: {model} failed, please check the logfile for details",
    5: "HMM failed, please check the logfile for details",
    6: "Models completed, script failed at validation"
}

peakRamMb = 0
logFile = None
startTime = time.time()
startDateTime = time.strftime("%Y-%m-%d %H:%M:%S")

def writeLogStart(path, stock):
    """Begin run-level logging and tracking."""
    global logFile

    updatePeakRam()

    outPath = Path(path)
    outPath.mkdir(parents=True, exist_ok=True)

    logFile = outPath / "logFile.txt"
    
    logLine = (
        f"{'=' * 85}\n"
        f"Stock: {stock}\n"
        f"GPR data downloaded from https://www.matteoiacoviello.com/gpr.htm on {datetime.now():%Y-%m-%d}\n"
        f"Stock information sourced from Yahoo finance and stooq.pl on {datetime.now():%Y-%m-%d}\n"
        f"Start: {startDateTime}\n"
        f"{'=' * 85}\n"
    )

    with open(logFile, "w") as f:
        f.write(logLine)
        
def writeLogEnd(status, url=""):
    """Write final run metadata to the log file."""
    updatePeakRam()

    endTime = time.time()
    endDateTime = time.strftime("%Y-%m-%d %H:%M:%S")
    wallClock = endTime - startTime
    formatted = time.strftime("%H:%M:%S", time.gmtime(wallClock))
    statusMessage = getStatusMessage(status, url)
    

    logEnd = (
        f"{'=' * 85}\n"
        f"{'=' * 85}\n"
        f"End: {endDateTime}\n"
        f"Wall: {formatted} ({wallClock:.2f}s)\n"
        f"Peak RAM: {peakRamMb:.2f} MB\n"
        f"{'=' * 85}\n"
        f"Exit Status: {status} - {statusMessage}\n"
    )
    
    with open(logFile, "a") as f:
        f.write(logEnd)
    
def setUpLogging():
    """Configure Python logging to write to logFile."""
    logging.basicConfig(
        filename=logFile,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    logging.captureWarnings(True)
    warnings.filterwarnings("always")

def manualLogStart(process, subprocess):
    """Log the start time of a pipeline process."""
    logStart = datetime.now()
    logLine = (
        f"{'=' * 85}\n"
        f"Processs starting: {process} - {subprocess} at {logStart:%Y-%m-%d %H:%M:%S}\n"
        )
    
    with open(logFile, "a") as f:
        f.write(logLine)
        
    return logStart 

def manualLogEnd(logStart, process, subprocess):
    """Log the end and duration of a pipeline process."""
    end = datetime.now() - logStart
    logLine = (
        f"Process finishing: {process} - {subprocess}, total time: {str(end).split('.')[0]}\n"
        )
    
    with open(logFile, "a") as f:
        f.write(logLine)

def checkConnection(url):
    """Check connectivity to a URL and return a status code."""
    timeout = 5
    url = url if url else "http://www.google.com"

    try:
        requests.head(url, timeout=timeout)
        return 0, url
    except requests.ConnectionError:
        return 2, url
    except requests.Timeout:
        return 3, url
    
def getStatusMessage(status, url=""):
    template = statusMapper.get(status, "Unknown status")
    return template.format(url=url)

def startRamMonitor(interval=5):
    """Start a daemon thread that monitors peak RAM usage."""
    def run():
        while True:
            updatePeakRam()
            time.sleep(interval)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    
def updatePeakRam():
    """Update the recorded peak RAM usage for the current process."""
    global peakRamMb

    process = psutil.Process()
    ramUsedMb = process.memory_info().rss / (1024 ** 2)

    if ramUsedMb > peakRamMb:
        peakRamMb = ramUsedMb

def makeOutPath(path, stock):
    """Build the output path for the current run."""
    date = datetime.now().strftime("%Y%m%d")
    stock = stock.replace(".", "")

    if path:
        outPath = Path(path)
    else:
        outPath = Path(f"./audit/macroAlphaForecast_{date}_{stock}/")
    
    return outPath

def configToJson(path):
    """Save serialisable config attributes to JSON for auditing."""
    outPath = Path(path)
    outPath.mkdir(parents=True, exist_ok=True)
    
    attributes = {
        key: value
        for key, value in config.__dict__.items()
        if not key.startswith("__") and isSerializable(value)
    }

    jsonConfig = {
        "module_name": config.__name__,
        "attributes": attributes
    }

    with open(f"{path}/config.json", "w") as f:
        json.dump(jsonConfig, f, indent=2)
        
def isSerializable(value):
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        return False

def marketsToCsv(path, dataFrames):
    outPath = Path(path /"marketData/")
    outPath.mkdir(parents=True, exist_ok=True)
    
    for name, df in dataFrames.items():
        filePath = os.path.join(outPath, f"{name}.csv")
        df.to_csv(filePath, index=False)
    
def saveEnResults(path, results, featureName, model):
    outPath = Path(path) / f"models/{model}/{featureName}/" 
    outPath.mkdir(parents=True, exist_ok=True)
    
    results.to_csv(outPath / "elasticNetResults.csv")
    
def saveHmmResults(path, hmmData, hmmResults, featureName): 
    outPath = Path(path) / f"models/hiddenMarkovModel/{featureName}"
    outPath.mkdir(parents=True, exist_ok=True)
    
    hmmData.to_csv(os.path.join(outPath / "hmmData.csv"))
    
    for name, obj in hmmResults.items():
        if name == "featureName":
            continue
        obj.to_csv(os.path.join(outPath,f"{name}.csv"))

def saveValidation(path, modelName, featureName, metrics, stock):
    outPath = Path(path) / f"validation/model/{modelName}/{featureName}"
    outPath.mkdir(parents=True, exist_ok=True)
    
    signalStr = metrics["signal"].to_string()
    
    metrics["validationDf"].to_csv(outPath / "validationData.csv")
    
    validationMetrics = (
        f"Stock: {stock}\n"
        f"Model: {modelName}\n"
        f"Feature: {featureName}\n"
        f"Total Return: {metrics['totalReturn']}\n" 
        f"Buy Hold return: {metrics['buyHoldReturn']}\n"
        f"Sharpe ratio: {metrics['sharpe']}\n"
        f"Hit Rate: {metrics['hitRate']}\n"
        f"Max drawdown: {metrics['maxDrawdown']}\n"
        f"Signal distribution:\n{signalStr}\n"
        )
    
    with open(outPath / "validationMetrics.txt", "w") as f:
            f.write(validationMetrics)
            
def savePlot(path, fig, parent, modelName, featureName, name):
    outPath = Path(path) / f"validation/{parent}/{modelName}/{featureName}"
    outPath.mkdir(parents=True, exist_ok=True)
    
    fig.write_html(outPath / name)

