"""
Quant Challenge 2025

Algorithmic strategy template with risk management.
"""

from enum import Enum
from typing import Optional
from collections import deque
import numpy as np


class Side(Enum):
    BUY = 0
    SELL = 1

class Ticker(Enum):
    # TEAM_A (home team)
    TEAM_A = 0

def place_market_order(side: Side, ticker: Ticker, quantity: float) -> None:
    """Place a market order."""
    print(f"--- Placing Market Order: {side.name} {quantity} {ticker.name} ---")
    return

def place_limit_order(side: Side, ticker: Ticker, quantity: float, price: float, ioc: bool = False) -> int:
    """Place a limit order."""
    return 0

def cancel_order(ticker: Ticker, order_id: int) -> bool:
    """Cancel an order."""
    return 0

class Strategy:
    """Template for a strategy."""

    def reset_state(self) -> None:
        """Reset the state of the strategy to the start of game position."""
        # --- Parameters ---
        self.trade_size = 5.0 # MODIFIED: Smaller trade size to reduce risk
        self.fast_ma_period = 10
        self.slow_ma_period = 40
        self.bb_period = 20
        self.bb_std_dev = 2.0

        # --- NEW: Risk Management Parameters ---
        self.stop_loss_pct = 0.02  # Exit if a trade loses 2%
        self.profit_target_pct = 0.03 # Exit if a trade gains 3%

        # --- State Variables ---
        self.prices = deque(maxlen=self.slow_ma_period)
        self.position = 0.0
        self.entry_price = None # NEW: To track the price of our current position
        self.trend_regime = "NONE"
        
    def __init__(self) -> None:
        """Your initialization code goes here."""
        self.reset_state()

    def on_trade_update(
        self, ticker: Ticker, side: Side, quantity: float, price: float
    ) -> None:
        """Called whenever two orders match. Contains all trading logic."""
        self.prices.append(price)

        # --- NEW: Check for Stop-Loss or Profit-Target FIRST ---
        if self.position != 0 and self.entry_price is not None:
            # Check for LONG position exits
            if self.position > 0:
                if price <= self.entry_price * (1 - self.stop_loss_pct):
                    print(f"STOP-LOSS triggered for long position at {price:.2f}")
                    place_market_order(Side.SELL, ticker, abs(self.position))
                    return # Exit to avoid placing another trade immediately
                if price >= self.entry_price * (1 + self.profit_target_pct):
                    print(f"PROFIT-TARGET triggered for long position at {price:.2f}")
                    place_market_order(Side.SELL, ticker, abs(self.position))
                    return
            # Check for SHORT position exits
            else: # self.position < 0
                if price >= self.entry_price * (1 + self.stop_loss_pct):
                    print(f"STOP-LOSS triggered for short position at {price:.2f}")
                    place_market_order(Side.BUY, ticker, abs(self.position))
                    return
                if price <= self.entry_price * (1 - self.profit_target_pct):
                    print(f"PROFIT-TARGET triggered for short position at {price:.2f}")
                    place_market_order(Side.BUY, ticker, abs(self.position))
                    return

        # --- Indicator and Entry Logic ---
        # Wait for enough data to calculate all indicators
        if len(self.prices) < self.slow_ma_period:
            return

        # Calculate Indicators
        fast_ma = np.mean(list(self.prices)[-self.fast_ma_period:])
        slow_ma = np.mean(self.prices)
        bb_prices = list(self.prices)[-self.bb_period:]
        bb_middle = np.mean(bb_prices)
        bb_std = np.std(bb_prices)
        bb_upper = bb_middle + self.bb_std_dev * bb_std
        bb_lower = bb_middle - self.bb_std_dev * bb_std

        # Determine Trend Regime
        previous_regime = self.trend_regime
        if fast_ma > slow_ma:
            self.trend_regime = "UP"
        else:
            self.trend_regime = "DOWN"

        # Exit Logic based on Trend Reversal
        if self.position > 0 and self.trend_regime == "DOWN" and previous_regime == "UP":
            place_market_order(Side.SELL, ticker, abs(self.position))
        elif self.position < 0 and self.trend_regime == "UP" and previous_regime == "DOWN":
            place_market_order(Side.BUY, ticker, abs(self.position))

        # Entry Logic: Only enter if we have no position
        if self.position == 0:
            if self.trend_regime == "UP" and price <= bb_lower:
                place_market_order(Side.BUY, ticker, self.trade_size)
            elif self.trend_regime == "DOWN" and price >= bb_upper:
                place_market_order(Side.SELL, ticker, self.trade_size)

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
        """Called whenever one of your orders is filled to track position and entry price."""
        # Record entry price ONLY on the trade that opens a position
        if self.position == 0:
            self.entry_price = price
        
        # Update position size
        if side == Side.BUY:
            self.position += quantity
        else:
            self.position -= quantity
        
        # If the position is now closed, reset the entry price
        if self.position == 0:
            self.entry_price = None

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
        """This function must accept all 13 arguments, even if they are not used."""
        if event_type == "END_GAME":
            self.reset_state()
            return