"""
Quant Challenge 2025

Algorithmic strategy template with a zero-profit reversal and profit preservation rule.
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
        self.trade_size = 5.0
        self.fast_ma_period = 10
        self.slow_ma_period = 40
        self.bb_period = 20
        self.bb_std_dev = 2.0
        self.stop_loss_pct = 0.02
        self.profit_target_pct = 0.03

        # --- NEW: Profit Preservation Parameters ---
        self.initial_capital = 1000000.0 # Example starting capital
        self.profit_threshold = 200000.0 # $200K profit target
        self.defensive_size_multiplier = 0.00001 # 0.001%

        # --- State Variables ---
        self.prices = deque(maxlen=self.slow_ma_period)
        self.position = 0.0
        self.entry_price = None
        self.trend_regime = "NONE"
        self.trade_was_profitable = False
        self.current_capital = self.initial_capital # NEW: Track capital
        self.defensive_mode_activated = False # NEW: Flag for profit preservation

    def __init__(self) -> None:
        """Your initialization code goes here."""
        self.reset_state()

    def on_trade_update(
        self, ticker: Ticker, side: Side, quantity: float, price: float
    ) -> None:
        """Called whenever two orders match. Contains all trading logic."""
        self.prices.append(price)

        # --- HIGHEST PRIORITY: Zero-Profit Reversal Logic ---
        if self.position != 0 and self.entry_price is not None:
            is_currently_profitable = (self.position > 0 and price > self.entry_price) or \
                                      (self.position < 0 and price < self.entry_price)
            if is_currently_profitable:
                self.trade_was_profitable = True
            if self.trade_was_profitable and not is_currently_profitable:
                reversal_side = Side.SELL if self.position > 0 else Side.BUY
                reversal_quantity = abs(self.position) * 2
                place_market_order(reversal_side, ticker, reversal_quantity)
                return

        # --- Second Priority: Stop-Loss or Profit-Target ---
        if self.position != 0 and self.entry_price is not None:
            if self.position > 0: # Long
                if price <= self.entry_price * (1 - self.stop_loss_pct) or \
                   price >= self.entry_price * (1 + self.profit_target_pct):
                    place_market_order(Side.SELL, ticker, abs(self.position))
                    return
            else: # Short
                if price >= self.entry_price * (1 + self.stop_loss_pct) or \
                   price <= self.entry_price * (1 - self.profit_target_pct):
                    place_market_order(Side.BUY, ticker, abs(self.position))
                    return

        # --- Third Priority: Indicator and Standard Entry/Exit Logic ---
        if len(self.prices) < self.slow_ma_period:
            return

        fast_ma = np.mean(list(self.prices)[-self.fast_ma_period:])
        slow_ma = np.mean(self.prices)
        bb_prices = list(self.prices)[-self.bb_period:]
        bb_middle = np.mean(bb_prices)
        bb_std = np.std(bb_prices)
        bb_upper = bb_middle + self.bb_std_dev * bb_std
        bb_lower = bb_middle - self.bb_std_dev * bb_std

        previous_regime = self.trend_regime
        if fast_ma > slow_ma:
            self.trend_regime = "UP"
        else:
            self.trend_regime = "DOWN"

        if self.position > 0 and self.trend_regime == "DOWN" and previous_regime == "UP":
            place_market_order(Side.SELL, ticker, abs(self.position))
        elif self.position < 0 and self.trend_regime == "UP" and previous_regime == "DOWN":
            place_market_order(Side.BUY, ticker, abs(self.position))

        # --- Entry Logic with NEW Profit Preservation Check ---
        if self.position == 0:
            # Determine the trade size to use
            current_trade_size = self.trade_size
            if self.defensive_mode_activated:
                current_trade_size = self.trade_size * self.defensive_size_multiplier
                print(f"DEFENSIVE MODE: Using reduced trade size: {current_trade_size}")

            if self.trend_regime == "UP" and price <= bb_lower:
                place_market_order(Side.BUY, ticker, current_trade_size)
            elif self.trend_regime == "DOWN" and price >= bb_upper:
                place_market_order(Side.SELL, ticker, current_trade_size)

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
        capital_remaining: float, # This value is now used
    ) -> None:
        """Called whenever an order is filled. Crucial for updating state and P&L."""
        
        # --- Update Capital (simplified: assumes no fees/slippage in this var) ---
        # A more robust P&L would calculate based on actual fills
        cost = price * quantity
        if side == Side.BUY:
            self.current_capital -= cost
        else:
            self.current_capital += cost
            
        # --- Check for Profit Threshold ---
        # P&L = current cash + value of current holdings - initial cash
        pnl = (self.current_capital + (self.position * price)) - self.initial_capital
        if pnl >= self.profit_threshold:
            if not self.defensive_mode_activated:
                print(f"SUCCESS: P&L of {pnl:.2f} crossed threshold of {self.profit_threshold}. Activating defensive mode.")
                self.defensive_mode_activated = True

        # --- Standard Position and State Management ---
        original_position = self.position
        if side == Side.BUY:
            self.position += quantity
        else:
            self.position -= quantity
        
        if (original_position > 0 and self.position < 0) or \
           (original_position < 0 and self.position > 0):
            self.entry_price = price
            self.trade_was_profitable = False
        elif original_position == 0 and self.position != 0:
            self.entry_price = price
            self.trade_was_profitable = False
        elif self.position == 0:
            self.entry_price = None
            self.trade_was_profitable = False

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