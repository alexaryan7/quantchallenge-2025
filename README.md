# QuantChallenge 2025

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Rank](https://img.shields.io/badge/Rank-Top_6%25_(91/1417)-brightgreen.svg)

---

## Overview

This repository contains the complete codebase and methodology for my participation in the **QuantChallenge 2025**, a two-phase competition focused on quantitative research and live algorithmic trading. The project showcases an end-to-end workflow, starting from initial data analysis and progressing to a sophisticated, machine learning-driven trading bot.

The final result was a **Top 6% finish**, ranking 91 out of 1,417 participants.

### Key Achievements:
* **Final Rank**: **91 / 1,417 (Top 6%)**
* **Research R² Score**: **0.65** (vs. the winning score of 0.71)
* **Trading Simulation**: Designed and tested multiple strategies, with one achieving a **simulated 300% return** ($400k profit on a $100k budget) in a single backtest run.

---

## 2. The Research Phase: Building the Predictive Model

The first phase involved building a model to predict two target variables, `Y1` and `Y2`, from a noisy time-series dataset. The approach was an iterative process of increasing complexity and robustness, starting from a simple baseline and culminating in a state-of-the-art ensemble.

### 2.1. Model Architecture
The final, highest-scoring model was a **stacked ensemble** designed to maximize predictive accuracy while controlling for overfitting.
* **Base Models**: A diverse set of five models including `LightGBM`, `XGBoost`, `CatBoost`, an `MLPRegressor` (Neural Network), and `Ridge` regression.
* **Meta-Model**: An `ElasticNet` model was trained on the out-of-fold predictions from the base layer to produce the final forecast.
* **Robust Validation**: Out-of-fold (OOF) predictions were generated using a `TimeSeriesSplit` walk-forward validation loop, ensuring the model was validated on unseen "future" data at every step.

### 2.2. Research Model Performance

The following chart illustrates the iterative journey of model improvement, tracking the public R² score as we progressed from a simple baseline to the final, complex pipeline. The systematic application of feature engineering, ensembling, and hyperparameter tuning resulted in a significant performance increase.

| Rank | Model / Strategy         | Public R² Score |
|------|---------------------------|-----------------|
|1     | Final Boosted Pipeline    | **0.6502**      |
|2     | Optuna-Tuned Stack        | 0.5920          |
|3     | 4-Model Stack             | 0.5801          |
| 4    | Grandmaster Script        | 0.5773          |
| 5    | Simple Stacked Ensemble   | 0.5766          |
| 6    | Complex 5-Model Stack     | 0.5706          |
| 7    | Averaged Ensemble         | 0.5373          |
| 8    | Single LightGBM Model     | 0.4910          |
| 9    | Simple Linear Model       | 0.4487          |

---

## 3. The Live Trading Phase: Strategy Implementation

The second phase required designing and deploying algorithms in a **5-minute simulated market environment**. This involved testing a wide range of strategies to explore different sources of "alpha" (predictive signals).

The **MVA-MR (Moving Average - Mean Reversion)** strategy proved to be exceptionally effective in this environment, generating a **simulated profit of $400,000** on a $100,000 budget in a single run.

The full list of strategies developed and tested includes:
* Momentum
* Pairs Trading (Statistical Arbitrage)
* MVA-MR (Moving Average - Mean Reversion)
* Parabolic SAR
* Trend-Filtered Strategies ("Profit Preservation")
* Custom Reversal Algorithms
* Hybrid Strategies (e.g., Game Momentum + Parabolic SAR)

For the complete implementation of these algorithms, please refer to the code in the `trading_bots/` directory of this repository.

---

## 4. Repository Structure

.
├── research/
│   ├── quant_challenge_boost_to_0_7.py   # The final, high-performance research script (R² 0.65).
│   └── r2_scores_progression.png         # The chart visualizing R² score improvements.
│
├── trading_bots/
│   ├── bot_ml_forecasting.py             # The final, self-contained ML-driven bot.
│   ├── bot_pairs_trading.py              # The Pairs Trading strategy bot.
│   └── bot_mva_mr.py                     # The highly profitable MVA-MR strategy bot.
│
├── data/
│   ├── train.csv                         # The training dataset.
│   └── test.csv                          # The test dataset.
│
└── README.md                             # This file.


---

## 5. Key Learnings

This project was a deep dive into the end-to-end quantitative workflow, strengthening my expertise in:
* **Quantitative Research**: Advanced feature engineering, time-series analysis, and robust cross-validation.
* **Machine Learning**: Building, tuning, and deploying complex stacking ensembles (LightGBM, XGBoost, CatBoost).
* **Strategy Design & Backtesting**: Implementing and testing a diverse portfolio of trading paradigms, from predictive ML to statistical arbitrage and mean-reversion.
* **Deployment Constraints**: Solving real-world deployment challenges like file size limits by 
