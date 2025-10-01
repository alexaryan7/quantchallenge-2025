# QuantChallenge 2025

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Rank](https://img.shields.io/badge/Rank-Top_6%25_(91/1417)-brightgreen.svg)

---

## 1. Project Overview

This repository contains the complete codebase and methodology for my participation in the **QuantChallenge 2025**, a two-phase competition focused on quantitative research and live algorithmic trading. The project showcases an end-to-end workflow, starting from initial data analysis and progressing to a sophisticated, machine learning-driven trading bot.

The final result was a **Top 6% finish**, ranking 91 out of 1,417 participants.

### Key Achievements:
* **Final Rank**: **91 / 1,417 (Top 6%)**
* **Research R² Score**: **0.65** (vs. the winning score of 0.71)
* **Trading Simulation**: Designed and tested multiple strategies, with one achieving a **simulated 300% return** ($400k profit on a $100k budget) in a single backtest run.

---

## 2. The Research Phase: Building the Predictive Model

The first phase of the competition involved building a model to predict two target variables, `Y1` and `Y2`, from a noisy time-series dataset. The R² score was the primary metric. My approach was an iterative process of increasing complexity and robustness.

### 2.1. Initial Baseline & Feature Engineering
Started with simple models and quickly progressed to a comprehensive feature engineering pipeline. This included:
* Time-series features: Lags, deltas, and ratios.
* Rolling statistics: Moving averages and standard deviations over multiple windows (5, 10, 30).
* Volatility measures: Exponentially Weighted Means (EWM) and rolling standard deviations.
* Advanced features: Pairwise interaction terms, rolling correlations, and "market regime" detection using K-Means clustering.

### 2.2. Ensemble Stacking (The R² 0.65 Breakthrough)
The core of the research model is a **stacked ensemble**. This approach combines the predictions of several different base models, feeding them into a final meta-model that learns to find the optimal blend.
* **Base Models**: A diverse set of models including `LightGBM`, `XGBoost`, `CatBoost`, `Ridge`, and an `MLPRegressor` (Neural Network).
* **Meta-Model**: An `ElasticNet` model was used to weigh the predictions from the base layer.
* **Robust Validation**: To prevent overfitting, out-of-fold (OOF) predictions for the meta-model were generated using a manual `TimeSeriesSplit` walk-forward validation loop.

### 2.3. Hyperparameter Tuning with Optuna
To maximize the performance of the most influential models, I used **Optuna** for systematic and robust hyperparameter optimization. The objective function was designed to maximize the average R² score across a 5-fold `TimeSeriesSplit`, ensuring the chosen parameters would generalize well to unseen data.

---

## 3. The Live Trading Phase: Strategy Implementation

The second phase required deploying a trading algorithm in a live, simulated market. The key challenge was a **100kb single `.py` file submission limit**, which prohibited the use of external `.joblib` model files.

### 3.1. Solution 1: The ML-Forecasting Bot (Model "Baking")
To overcome the file limit, I developed a two-stage deployment process:
1.  **"Baking" Script**: A script that trains a single, highly-tuned `LightGBM` model and then exports the model's entire internal structure (trees, splits, and leaves) into a large Python dictionary.
2.  **Deployment Script**: The final trading bot loads this dictionary and reconstructs the trained model in memory at runtime. This "bakes" the model's intelligence directly into the script.

The bot's logic involves:
* Maintaining a live history of market data.
* Running the feature engineering pipeline on the fly.
* Handling recursive features (like `Y1_lag1`) by feeding its own previous prediction back into the feature set.
* Executing trades when the model's prediction crosses a predefined threshold above or below the current market price.

### 3.2. Solution 2: The Pairs Trading (Statistical Arbitrage) Bot
As an alternative strategy, I also developed a complete Pairs Trading bot.
* **Logic**: It identifies a highly correlated pair of features from the dataset (e.g., 'C' and 'A'), calculates the Z-score of their price spread, and executes trades when the spread diverges significantly from its historical mean, betting on its reversion.

---

## 4. Repository Structure

.
├── research/
│   ├── quant_challenge_boost_to_0_7.py   # The final, high-performance research script (R² 0.65).
│   └── train_and_save_model.py           # Script to train and save models for local testing.
│
├── trading_bots/
│   ├── bot_ml_forecasting.py             # The final, self-contained ML-driven bot with the embedded model.
│   └── bot_pairs_trading.py              # The complete Pairs Trading strategy bot.
│
├── data/
│   ├── train.csv                         # The training dataset.
│   └── test.csv                          # The test dataset.
│
└── README.md                             # This file.


---

## 5. How to Use

### Research Phase
1.  Navigate to the `research/` directory.
2.  Run `quant_challenge_boost_to_0_7.py` to replicate the R² 0.65 model and generate a submission file (`submission_boosted_pipeline.csv`).

### Deployment for Live Trading
1.  First, run a "baking" script (like the one developed in our chat) to train a lean `LightGBM` model and print its structure as a Python dictionary.
2.  Copy the entire `model_dict` and `features_to_use` output.
3.  Paste this output into the placeholder section at the top of `trading_bots/bot_ml_forecasting.py`.
4.  The `bot_ml_forecasting.py` file is now a complete, self-contained algorithm ready for submission.

---

## 6. Key Learnings

This project was a deep dive into the end-to-end quantitative workflow, strengthening my expertise in:
* **Quantitative Research**: Advanced feature engineering, time-series analysis, and robust validation techniques.
* **Machine Learning**: Building, tuning (Optuna), and deploying complex stacking ensembles (LightGBM, XGBoost).
* **Strategy Design**: Implementing and testing multiple trading paradigms, from predictive ML to statistical arbitrage.
* **Deployment Constraints**: Solving real-world deployment challenges like file size limits by "baking" models into scripts.

---
*This README was generated with the assistance of an AI model
