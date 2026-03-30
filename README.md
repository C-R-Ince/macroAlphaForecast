# Macro Alpha Forecast
## A regime-aware, time-decayed framework for selective extraction of high-conviction macro signals

This project investigates whether financial markets exhibit regime-dependent behaviour, where relationships within regimes remain approximately linear but differ across regimes.

To explore this, a walk-forward modelling pipeline was developed combining Hidden Markov Models (for latent regime detection) with Elastic Net regression (for regularised linear modelling). The objective was to assess whether regime-aware linear models can extract signal from noisy macroeconomic features such as Geopolitical Risk (GPR).

The focus of the project is on understanding model behaviour and signal structure, rather than maximising raw predictive performance.

Testing across a diversified set of stocks shows that predictive power is concentrated in a small subset of high-confidence signals. The model performs best as a selective filter, extracting alpha during informative regimes while avoiding low-signal periods.

However, the model does not act as a consistent return forecaster and may underperform buy-and-hold strategies in strongly trending or macro-insensitive assets.

These results suggest that macro signals are not universally predictive, but can provide meaningful edge when conditioned on regime and signal strength.

## TL;DR

* Regime-aware models outperform baseline in risk-adjusted terms  
* Signal is sparse and concentrated in high-confidence periods  
* Model acts as a **filter**, not a continuous predictor

## Research Hypothesis
**H₀ (Null Hypothesis):**
Incorporating regime information does not improve predictive performance or materially change the relationships learned by the model.
**H₁ (Alternative Hypothesis):** 
Markets exhibit regime-dependent behaviour, while relationships within regimes are approximately linear. Therefore, incorporating regime information via a Hidden Markov Model into a regularised linear model (Elastic Net) should improve predictive performance and enable the extraction of signal from noisy macroeconomic features.


## Overview

This project builds a walk-forward modelling pipeline to predict future returns using:

* Time-decay Baseline Elastic Net
* Walk-forward Hidden-Markov Model
* Time-decay Regime-feature Elastic Nets
* Time-decay Regime-Probability Elastic Nets

The goal is to evaluate whether incorporating latent market regimes improves predictive power and trading performance.


## Pipeline
```
		   Data sourced (GPR, config stock, user input stock)
						|
						V
		   Data handling and feature generation	
		 		|					|
		 		V					|
		   Baseline 				|
		   ElasticNetCV				|
		   (Time-decay)				|
		 		|					V
		 		|	Walk-forward Hmm of GPR and config stock
		 		|		 |						|
		 		|		 V						V
		 		|	Regime Feature		  Regime Probability
		 		|	ElasticNetCV 		  ElasticNetCV
		 		|	(Time-decay)		  (Time-decay)
				|		|						|
				V		V						|
			Model Validation and Evaluation		V
			(Total Return, Sharpe ratio, Hit Rate, and Drawdown)
```			
	
				
## Scripts

| Script            | Description                                                         				  |
| :---------------- | :---------------------------------------------------------------------------------- |
| `main.py`		    | Entry point for running the full pipeline                            				  |
| `marketFeat.py`   | Hybrid ETL pipeline that prioritizes yfinance with Stooq as a resilient fallback.   |
| `model.py`        | Implements Elastic Net models, HMM, and regime logic              				  |
| `validation.py`   | Computes performance metrics and backtests                           				  |
| `config.py`       | Fully configurable file for model parameters and windows             				  |
| `log.py`          | Handles logging, output saving, and audit file creation              				  |


## Methods

**Model:** 
Time-decay Elastic Net augmented with HMM regime probabilities or regime-specific interactions

**Evaluation metrics:**
Total Return
Sharpe Ratio
Hit Rate
Maximum Drawdown

**Feature exploration:**
Macroeconomic indicators, GPR, commodities, and sector ETFs as explanatory variables for a target stock


## Installation

```bash
pip install -r requirements.txt
```

```txt
hmmlearn==0.3.3
joblib==1.5.3
numpy==2.4.3
pandas==3.0.1
psutil==7.2.2
python-dateutil==2.9.0.post0
scikit-learn==1.8.0
scipy==1.17.1
six==1.17.0
threadpoolctl==3.6.0
```


## Project Structure
```txt
src/
│
├── main.py
├── marketFeat.py           # Feature engineering 
├── model.py	            # Elastic Net + HMM models
├── validation.py           # Backtesting + metrics
├── config.py               # Parameters
└── log.py                  # Logging / outputs
```


## Feature Sets

Features at time \( t \) are constructed using only past returns, ensuring temporal causality and preventing information leakage:

| Feature 	   		| Equation 									 	    | Description 					     |
| :---------------- |:--------------------------------------------------|:-----------------------------------|
| `return` 	  	    | r_t = log(P_t) - log(P_{t-1})  				    | Log return at time t               |
| `rolling{WINDOW}` | (1/WINDOW) * sum_{i=1..W} r_{t-i}   		   	    | Average return over [t-W, t-1]     |
| `lag1` 	   		| r_{t-1} 									 	    | Log return lagged by one period    |
| `lag2` 	   		| r_{t-2} 										    | Log return lagged by two periods   |
| `vol` 	   		| sqrt((1/WINDOW) * sum_{i=1..W} (r_{t-i} - r̄_t)^2) | Rolling volatility over [t-W, t-1] |

The model predicts r_{t+1}, ensuring no look-ahead bias.


## Example Usage
```bash
pip install -r requirements.txt
python src/main.py --ticker <yFinance_ticker>
```


## Optional Arguments

```bash
--output path/to/output/  			  			  	# Define output directory for the project
```


## Output Structure (assuming default file path)
```txt
./audit/
    └─── macroAlphaForecast_YYYYMMDD_{ticker}/
			│
			├─── marketData /						# Data generated during preprocessing
			│		├─── fullData.csv 				# Full dataset used in modelling (audit trail)
			│ 		├─── marketData.csv				# Market + macro features
			│ 		└─── targetData.csv				# Target stock + engineered features
			├─── models /							# Model outputs (by model type, then feature set)
			│       ├── baselineElasticNet /		
			│       ├── elasticNetHmmProb /
			│       ├── elasticNetRegimeSpec /
			│		└── hiddenMarkovModel /
			├─── validation /					 	# Backtest results and evaluation metrics
			│       ├── model /						# Validation metrics per model (Backtesting, sharpe, 
			│       ├── feature /					# Cumulative returns calculated per feature	
			│		└── sharpeHeatmap.html			# Heatmap displaying sharpe ratio, model x feature
			├─── logFile.txt						# Log file storing run date and time, wall clock time, error messages, and exit status
			└─── config.json						# JSON file that stores all configurable information at the time of running
```


## Project Goals

* Evaluate macroeconomic variables and Geopolitical Risk (GPR) as forecasting signals for user-selected equities
* Investigate whether walk-forward Hidden Markov Models (HMMs) enhance predictive performance of time-decay Elastic Net models in the presence of multicollinearity

## Key Findings

* **Model performance is asset-dependent.**  
  The framework performs strongly on macro-sensitive equities (e.g. XOM), fails on idiosyncratic growth stocks (e.g. AAPL), and shows mixed results on partially macro-driven assets (e.g. CVX, BA.L). This suggests that macro signals are only informative when the underlying asset exhibits sufficient macro sensitivity.

* **Predictive power is concentrated in high-confidence signals.**  
  Strong signals drive the majority of performance, while weaker signals behave as noise. Lowering the signal threshold increases trading frequency but degrades both Sharpe ratio and total return, confirming that the model is most effective as a selective filter rather than a continuous predictor.

* **Regime-aware modelling improves risk-adjusted performance.**  
  Hidden Markov Model (HMM) conditioning enhances Elastic Net performance relative to baseline models. The effectiveness depends on regime specification, with simpler regime structures (e.g. two regimes) providing more stable and interpretable results than higher-complexity alternatives.

* **Time-decay is critical for signal extraction.**  
  Increasing time-decay weighting improves performance by prioritising recent data, indicating that both market state and recency are essential for extracting signal from macro features.

* **The model acts as a risk filter rather than a return forecaster.**  
  Performance is primarily driven by avoiding adverse periods rather than capturing full upside trends. This leads to improved Sharpe ratios but potential underperformance versus buy-and-hold in strongly trending assets.

* **Regime complexity interacts with volatility representation.**  
  Increasing the number of regimes, with sufficient historic data, can endogenise volatility within regime classification, reducing its effectiveness as an independent feature and shifting the model toward a regime-conditioned linear structure dominated by directional return signals.

Together, these results indicate that macro signals do not provide universal predictive power, but can offer meaningful edge when conditioned on regime, recency, and signal strength.


## Limitations

* Relies on free and open source data for stock information that is subject to disruptions
* Limited feature set may restrict model expressiveness and predictive power
* Macroeconomic and GPR features are inherently noisy and subject to revisions
* Hidden Markov Models assume a fixed number of regimes, which may not reflect evolving market dynamics
* Regimes are treated as discrete states, whereas real-world market behaviour is often continuous

## Future Improvements

* Add unique run IDs and stop overwriting previous runs
* Add threshold-based exits of models due to accumulated warnings
* Implement Bayesian priors to inform more appropriate regime switching
* Extend the model to a Markov-switching state-space framework, using Kalman filtering within regimes and EM for estimation of regime-specific parameters and transition dynamics.
* Add further data sources as fall backs (e.g. Alpha Finance)
* Expand feature set (e.g. momentum, credit spreads, alternative data)

## Notes

* This is a research project and does not constitute financial advice
* Results are sensitive to data choices, feature engineering, and modelling assumptions
* Backtest performance may not generalise to live trading conditions

The modelling framework combines:

* Regime detection via Hidden Markov Models (Hamilton, 1989)
* Regularised regression via Elastic Net (Zou & Hastie, 2005)
* Walk-forward validation following standard time-series forecasting practice (Hyndman & Athanasopoulos, 2018)

## References

* Caldara, D., & Iacoviello, M. (2022).  
  Measuring Geopolitical Risk. *American Economic Review*, 112(4), 1194–1225.

* Hamilton, J. D. (1989).  
  A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle.  
  *Econometrica*, 57(2), 357–384.

* Hyndman, R. J., & Athanasopoulos, G. (2018).  
  *Forecasting: Principles and Practice* (2nd ed.). OTexts.  
  https://otexts.com/fpp2/

* Zou, H., & Hastie, T. (2005).  
  Regularization and Variable Selection via the Elastic Net.  
  *Journal of the Royal Statistical Society: Series B*, 67(2), 301–320.

* Stooq.  
  Historical market data. https://stooq.pl/
  
* Yahoo finance.
  Historical market data. https://uk.finance.yahoo.com/
