"""
Quant Challenge 2025

XGBoost implementation inserted into the strategy template.
Patched to:
- avoid DataFrame fragmentation (concat new columns once)
- handle xgboost early stopping API differences
- replace deprecated fillna(method=...) with ffill/bfill
"""

import os
from enum import Enum
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.base import clone
import joblib

import xgboost as xgb

# ---------------------------
# Provided template enums / functions (kept)
# ---------------------------
class Side(Enum):
    BUY = 0
    SELL = 1

class Ticker(Enum):
    TEAM_A = 0

def place_market_order(side: Side, ticker: Ticker, quantity: float) -> None:
    return

def place_limit_order(side: Side, ticker: Ticker, quantity: float, price: float, ioc: bool = False) -> int:
    return 0

def cancel_order(ticker: Ticker, order_id: int) -> bool:
    return 0

# ---------------------------
# Helper: xgboost fit with robust early stopping
# ---------------------------
def xgb_fit_with_earlystop(model, X_tr, y_tr, X_val=None, y_val=None, sample_weight=None, early_rounds=100, verbose=False):
    """
    Fit XGBRegressor robustly across xgboost versions.
    - If X_val/y_val provided, tries to use early_stopping_rounds.
    - Tries common signatures, falls back to callbacks, and finally to no early stopping.
    """
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    # If validation provided, attempt early stopping styles
    if X_val is not None and y_val is not None:
        try:
            # Most common usage
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=early_rounds, verbose=verbose, **fit_kwargs)
            return model
        except TypeError:
            pass
        except Exception:
            # unknown error — try callback approach as next attempt
            pass

        try:
            # Try callback-based early stopping
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[xgb.callback.EarlyStopping(rounds=early_rounds)], verbose=verbose, **fit_kwargs)
            return model
        except Exception:
            # fallback to fitting without early stopping
            try:
                model.fit(X_tr, y_tr, verbose=verbose, **fit_kwargs)
                return model
            except Exception:
                # final fallback: try numpy arrays
                model.fit(X_tr.values, y_tr.values, verbose=verbose, **fit_kwargs)
                return model
    else:
        # No validation set: fit normally (with sample_weight if supported)
        try:
            model.fit(X_tr, y_tr, verbose=verbose, **fit_kwargs)
            return model
        except Exception:
            model.fit(X_tr.values, y_tr.values)
            return model

# ---------------------------
# Model / Strategy Implementation
# ---------------------------
class Strategy:
    """
    Strategy wrapper that trains XGBoost models for Y1 and Y2,
    exposes predict_row for event-driven calls, and can produce submission CSV.
    """

    def reset_state(self) -> None:
        """Reset internal state (not used heavily here)."""
        self.last_event_index = None

    def __init__(self) -> None:
        """Initialize strategy: load data, engineer features, train models."""
        self.reset_state()
        self.train_df = None
        self.test_df = None
        self.features: List[str] = []
        self.model_y1 = None
        self.model_y2 = None
        # hyperparameters (tuned reasonably for tabular/time-series)
        self.xgb_params = {
            "objective": "reg:squarederror",
            "n_estimators": 2000,
            "learning_rate": 0.03,
            "max_depth": 6,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "eval_metric": "rmse",
            "random_state": 42,
            "tree_method": "hist",
            "verbosity": 0
        }

        # load data and train
        self._load_data()
        self._engineer_features()
        self._train_models()

    # ---------------------------
    # Data loading & FE
    # ---------------------------
    def _load_data(self):
        """Loads train.csv and test.csv from current working directory."""
        if not os.path.exists("train.csv") or not os.path.exists("test.csv"):
            raise FileNotFoundError("train.csv and test.csv must be present in working directory.")
        self.train_df = pd.read_csv("train.csv").reset_index(drop=True)
        self.test_df = pd.read_csv("test.csv").reset_index(drop=True)
        # ensure time ordering
        if "time" in self.train_df.columns:
            self.train_df.sort_values("time", inplace=True)
            self.test_df.sort_values("time", inplace=True)
        self.train_df.reset_index(drop=True, inplace=True)
        self.test_df.reset_index(drop=True, inplace=True)

    def _engineer_features(self):
        """Create basic time-series features for columns A-N with batch concat to avoid fragmentation."""
        df_all = pd.concat([self.train_df.drop(columns=["Y1", "Y2"]), self.test_df], ignore_index=True)
        # identify A-N columns (exclude time, id if present)
        candidate_cols = [c for c in self.train_df.columns if c not in ("time", "Y1", "Y2", "id")]
        features = candidate_cols.copy()

        eps = 1e-8
        windows = [5, 10, 30]

        # Build new columns in a dict, then concat once to avoid fragmentation
        new_cols = {}
        for f in features:
            col = df_all[f]
            new_cols[f"{f}_lag1"] = col.shift(1)
            # delta uses the lag (NaN allowed for now)
            new_cols[f"{f}_delta1"] = col - new_cols[f"{f}_lag1"]
            new_cols[f"{f}_ratio1"] = col / (new_cols[f"{f}_lag1"] + eps)
            for w in windows:
                new_cols[f"{f}_rm_{w}"] = col.rolling(window=w).mean()
                new_cols[f"{f}_rs_{w}"] = col.rolling(window=w).std()
            new_cols[f"{f}_ewm10"] = col.ewm(span=10, adjust=False).mean()
            new_cols[f"{f}_expmean"] = col.expanding().mean()

        # concat new columns once
        new_df = pd.DataFrame(new_cols, index=df_all.index)
        df_all = pd.concat([df_all, new_df], axis=1)

        # Pairwise interactions for top variance features (compute on df_all copy, then concat)
        var_scores = {f: df_all[f].var() for f in features}
        top_feats = sorted(var_scores, key=var_scores.get, reverse=True)[:6]
        inter_cols = {}
        for i in range(len(top_feats)):
            for j in range(i+1, len(top_feats)):
                a, b = top_feats[i], top_feats[j]
                inter_cols[f"{a}_x_{b}"] = df_all[a] * df_all[b]
                inter_cols[f"{a}_div_{b}"] = df_all[a] / (df_all[b] + eps)
        inter_df = pd.DataFrame(inter_cols, index=df_all.index)
        df_all = pd.concat([df_all, inter_df], axis=1)

        # Rolling correlations
        corr_cols = {}
        if len(top_feats) >= 3:
            pairs = [(top_feats[0], top_feats[1]), (top_feats[0], top_feats[2])]
            for a, b in pairs:
                corr_cols[f"corr_{a}_{b}_30"] = df_all[a].rolling(30).corr(df_all[b])
        corr_df = pd.DataFrame(corr_cols, index=df_all.index)
        df_all = pd.concat([df_all, corr_df], axis=1)

        # KMeans regime clustering (fit on filled copy)
        km_features = features
        tmp = df_all[km_features].ffill().bfill().fillna(0)
        from sklearn.cluster import KMeans
        try:
            km = KMeans(n_clusters=8, random_state=42, n_init=10)
            df_all['regime'] = km.fit_predict(tmp)
        except Exception:
            # if clustering fails for any reason, fallback to zeros
            df_all['regime'] = 0

        # fill forward/backwards to clear NaNs from rolling ops
        df_all.bfill(inplace=True)
        df_all.ffill(inplace=True)

        # split back
        n_train = len(self.train_df)
        train_proc = df_all.iloc[:n_train].copy().reset_index(drop=True)
        test_proc = df_all.iloc[n_train:].copy().reset_index(drop=True)

        # attach targets back to train_proc
        train_proc["Y1"] = self.train_df["Y1"].values
        train_proc["Y2"] = self.train_df["Y2"].values

        # Add simple target lag features (only available in train)
        train_proc["Y1_lag1"] = train_proc["Y1"].shift(1)
        train_proc["Y2_lag1"] = train_proc["Y2"].shift(1)
        # replace any NaNs in target-lags conservatively
        train_proc["Y1_lag1"].bfill(inplace=True)
        train_proc["Y2_lag1"].bfill(inplace=True)

        # In test we will set Y_lag features recursively during prediction
        test_proc["Y1_lag1"] = np.nan
        test_proc["Y2_lag1"] = np.nan

        # set attributes
        self.train_proc = train_proc
        self.test_proc = test_proc
        # finalize feature list (exclude id/time/targets)
        exclude = set(["id", "time", "Y1", "Y2"])
        feat_cols = [c for c in train_proc.columns if c not in exclude]
        # ensure target lags included
        if "Y1_lag1" not in feat_cols:
            feat_cols.append("Y1_lag1")
        if "Y2_lag1" not in feat_cols:
            feat_cols.append("Y2_lag1")
        self.features = feat_cols

    # ---------------------------
    # Sample weighting helper
    # ---------------------------
    def _time_decay_weights(self, n, tau=8000.0):
        """
        Exponential decay weights: older rows get smaller weight.
        idx 0 = oldest, idx n-1 = newest -> weight highest for newest
        """
        idx = np.arange(n)
        w = np.exp(-(n - 1 - idx) / tau)
        return w

    # ---------------------------
    # Training
    # ---------------------------
    def _train_models(self):
        """
        Train XGB models for Y1 and Y2 using TimeSeriesSplit CV and early stopping.
        Stores final models in self.model_y1 and self.model_y2.
        """
        X = self.train_proc[self.features].reset_index(drop=True)
        y1 = self.train_proc["Y1"].reset_index(drop=True)
        y2 = self.train_proc["Y2"].reset_index(drop=True)
        n = len(X)
        weights = self._time_decay_weights(n, tau=8000.0)

        tscv = TimeSeriesSplit(n_splits=5)
        oof_y1 = np.zeros(n)
        oof_y2 = np.zeros(n)

        print("Starting walk-forward CV training for XGBoost (Y1 & Y2)")
        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
            print(f"Fold {fold} | train {len(tr_idx)} -> val {len(val_idx)}")
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y1_tr, y1_val = y1.iloc[tr_idx], y1.iloc[val_idx]
            y2_tr, y2_val = y2.iloc[tr_idx], y2.iloc[val_idx]
            w_tr = weights[tr_idx]

            # Y1 model
            model_y1_cv = xgb.XGBRegressor(**self.xgb_params)
            model_y1_cv = xgb_fit_with_earlystop(model_y1_cv, X_tr, y1_tr, X_val, y1_val, sample_weight=w_tr, early_rounds=100, verbose=False)
            preds_val_y1 = model_y1_cv.predict(X_val)
            oof_y1[val_idx] = preds_val_y1

            # Y2 model
            model_y2_cv = xgb.XGBRegressor(**self.xgb_params)
            model_y2_cv = xgb_fit_with_earlystop(model_y2_cv, X_tr, y2_tr, X_val, y2_val, sample_weight=w_tr, early_rounds=100, verbose=False)
            preds_val_y2 = model_y2_cv.predict(X_val)
            oof_y2[val_idx] = preds_val_y2

            # print fold R2s
            r2_y1 = r2_score(y1_val, preds_val_y1)
            r2_y2 = r2_score(y2_val, preds_val_y2)
            print(f" Fold {fold} R2 Y1: {r2_y1:.4f}, Y2: {r2_y2:.4f}")

        # overall OOF R2 (rough estimate)
        overall_r2_y1 = r2_score(y1, oof_y1)
        overall_r2_y2 = r2_score(y2, oof_y2)
        print(f"OOF (CV) R2 estimate -> Y1: {overall_r2_y1:.4f}, Y2: {overall_r2_y2:.4f}, Avg: {(overall_r2_y1+overall_r2_y2)/2:.4f}")

        # Fit final models on ALL data (with weights)
        print("Fitting final XGBoost models on full train set...")
        final_y1 = xgb.XGBRegressor(**self.xgb_params)
        final_y1 = xgb_fit_with_earlystop(final_y1, X, y1, X_val=None, y_val=None, sample_weight=weights, early_rounds=0, verbose=False)

        final_y2 = xgb.XGBRegressor(**self.xgb_params)
        final_y2 = xgb_fit_with_earlystop(final_y2, X, y2, X_val=None, y_val=None, sample_weight=weights, early_rounds=0, verbose=False)

        self.model_y1 = final_y1
        self.model_y2 = final_y2

        # persist models for later reuse
        try:
            joblib.dump(self.model_y1, "model_xgb_y1.pkl")
            joblib.dump(self.model_y2, "model_xgb_y2.pkl")
            print("Saved models: model_xgb_y1.pkl, model_xgb_y2.pkl")
        except Exception as e:
            print("Could not save models via joblib:", e)

    # ---------------------------
    # Prediction utilities
    # ---------------------------
    def predict_row(self, row: pd.Series) -> dict:
        """
        Predict Y1 and Y2 for a single row (row must contain all features used).
        Returns dict: {'Y1': val, 'Y2': val}
        """
        if self.model_y1 is None or self.model_y2 is None:
            raise RuntimeError("Models not trained/loaded.")
        X_row = row[self.features].values.reshape(1, -1)
        y1p = float(self.model_y1.predict(X_row)[0])
        y2p = float(self.model_y2.predict(X_row)[0])
        return {"Y1": y1p, "Y2": y2p}

    def produce_submission(self, output_filename: str = "submission_xgb_template.csv"):
        """
        Produce submission file by recursively predicting test rows.
        Uses last train Y values as seed for test Y_lag1 and updates recursively.
        """
        test_df_copy = self.test_proc.copy().reset_index(drop=True)
        last_y1 = float(self.train_proc["Y1"].iloc[-1])
        last_y2 = float(self.train_proc["Y2"].iloc[-1])

        # ensure any NA filled
        test_df_copy.ffill(inplace=True)
        test_df_copy.bfill(inplace=True)

        preds_y1 = []
        preds_y2 = []
        for i in range(len(test_df_copy)):
            # set lag features
            test_df_copy.at[i, "Y1_lag1"] = last_y1
            test_df_copy.at[i, "Y2_lag1"] = last_y2
            row = test_df_copy.loc[i, self.features]
            pred = self.predict_row(row)
            y1p, y2p = pred["Y1"], pred["Y2"]
            preds_y1.append(y1p)
            preds_y2.append(y2p)

            # update seeds: blend new prediction with last true to moderate drift
            last_y1 = 0.75 * y1p + 0.25 * last_y1
            last_y2 = 0.75 * y2p + 0.25 * last_y2

        # post-process: clip to train range
        y1_min, y1_max = float(self.train_proc["Y1"].min()), float(self.train_proc["Y1"].max())
        y2_min, y2_max = float(self.train_proc["Y2"].min()), float(self.train_proc["Y2"].max())

        preds_y1 = np.clip(preds_y1, y1_min, y1_max)
        preds_y2 = np.clip(preds_y2, y2_min, y2_max)

        submission = pd.DataFrame({
            "id": self.test_df["id"].values,
            "Y1": preds_y1,
            "Y2": preds_y2
        })
        submission.to_csv(output_filename, index=False)
        print(f"Wrote submission -> {output_filename}")
        return submission

    # ---------------------------
    # Event handlers (kept as in template)
    # ---------------------------
    def on_trade_update(
        self, ticker: Ticker, side: Side, quantity: float, price: float
    ) -> None:
        print(f"Python Trade update: {ticker} {side} {quantity} shares @ {price}")

    def on_orderbook_update(
        self, ticker: Ticker, side: Side, quantity: float, price: float
    ) -> None:
        pass

    def on_account_update(
        self,
        ticker: Ticker,
        side: Side,
        price: float,
        quantity: float,
        capital_remaining: float,
    ) -> None:
        pass

    def on_game_event_update(self,
                           event_type: str,
                           home_away: str,
                           home_score: int,
                           away_score: int,
                           player_name: Optional[str],
                           substituted_player_name: Optional[str],
                           shot_type: Optional[str],
                           assist_player: Optional[str],
                           rebound_type: Optional[str],
                           coordinate_x: Optional[float],
                           coordinate_y: Optional[float],
                           time_seconds: Optional[float]
        ) -> None:
        print(f"{event_type} {home_score} - {away_score}")
        if event_type == "END_GAME":
            self.reset_state()
            return

# # ---------------------------
# # Standalone run: train and write submission
# # ---------------------------
# if __name__ == "__main__":
#     strat = Strategy()
#     # produce submission (file saved)
#     submission_df = strat.produce_submission("submission_xgb_template.csv")
#     # quick preview
#     print(submission_df.head())
