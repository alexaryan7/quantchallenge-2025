"""
Quant Challenge 2025

Algorithmic strategy template
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
    return True

class Strategy:
    """Template for a strategy."""

    def reset_state(self) -> None:
        """Reset the state of the strategy to the start of game position."""
        # --- Parameters ---
        self.trade_size = 5.0
        self.stop_loss_pct = 0.04
        
        # Fundamental Parameters
        self.momentum_threshold = 2.0
        self.event_weights = {
            "SHOT_MADE_3PT": 3.0, "SHOT_MADE_2PT": 2.0, "SHOT_MADE_1PT": 1.0,
            "REBOUND_OFFENSIVE": 1.5, "TURNOVER": -2.0, "FOUL": -1.0,
        }

        # Technical Parameters (Parabolic SAR)
        self.sar_acceleration = 0.02
        self.sar_max_acceleration = 0.20

        # --- State Variables ---
        self.prices = deque(maxlen=100)
        self.highs = deque(maxlen=100)
        self.lows = deque(maxlen=100)
        
        self.position = 0.0
        self.entry_price = None
        self.momentum_score = 0.0
        
        self.sar = None
        self.extreme_point = None
        self.acceleration_factor = self.sar_acceleration
        self.is_uptrend = True

    def __init__(self) -> None:
        """Your initialization code goes here."""
        self.reset_state()

    def _update_sar(self):
        """Helper function to calculate the next Parabolic SAR value."""
        if len(self.prices) < 2: return
        if self.sar is None:
            self.sar, self.extreme_point = (self.lows[-2], self.highs[-1])
            return

        current_price = self.prices[-1]
        
        if self.is_uptrend:
            self.sar = min(self.sar + self.acceleration_factor * (self.extreme_point - self.sar), self.lows[-2], self.lows[-1])
        else:
            self.sar = max(self.sar - self.acceleration_factor * (self.sar - self.extreme_point), self.highs[-2], self.highs[-1])

        new_trend = None
        if self.is_uptrend and current_price < self.sar: new_trend = "DOWN"
        elif not self.is_uptrend and current_price > self.sar: new_trend = "UP"

        if new_trend:
            self.sar = self.extreme_point
            self.acceleration_factor = self.sar_acceleration
            if new_trend == "UP":
                self.is_uptrend, self.extreme_point = True, self.highs[-1]
            else:
                self.is_uptrend, self.extreme_point = False, self.lows[-1]
        else:
            if self.is_uptrend and self.highs[-1] > self.extreme_point:
                self.extreme_point = self.highs[-1]
                self.acceleration_factor = min(self.sar_max_acceleration, self.acceleration_factor + self.sar_acceleration)
            elif not self.is_uptrend and self.lows[-1] < self.extreme_point:
                self.extreme_point = self.lows[-1]
                self.acceleration_factor = min(self.sar_max_acceleration, self.acceleration_factor + self.sar_acceleration)

    def on_trade_update(
        self, ticker: Ticker, side: Side, quantity: float, price: float
    ) -> None:
        """Called whenever two orders match. Handles technical logic."""
        self.prices.append(price)
        self.highs.append(price)
        self.lows.append(price)

        if self.position != 0 and self.entry_price is not None:
            if self.position > 0 and price <= self.entry_price * (1 - self.stop_loss_pct):
                place_market_order(Side.SELL, ticker, abs(self.position))
                return
            if self.position < 0 and price >= self.entry_price * (1 + self.stop_loss_pct):
                place_market_order(Side.BUY, ticker, abs(self.position))
                return

        was_uptrend = self.is_uptrend
        self._update_sar()
        if self.sar is None: return

        if self.momentum_score > self.momentum_threshold and self.position == 0:
            if not was_uptrend and self.is_uptrend:
                place_market_order(Side.BUY, ticker, self.trade_size)

        elif self.momentum_score < -self.momentum_threshold and self.position == 0:
            if was_uptrend and not self.is_uptrend:
                place_market_order(Side.SELL, ticker, self.trade_size)
        
        if self.position > 0 and not self.is_uptrend:
            place_market_order(Side.SELL, ticker, abs(self.position))
        elif self.position < 0 and self.is_uptrend:
            place_market_order(Side.BUY, ticker, abs(self.position))

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
        """Called whenever one of your orders is filled."""
        if self.position == 0: self.entry_price = price
        if side == Side.BUY: self.position += quantity
        else: self.position -= quantity
        if abs(self.position) < 1e-9:
            self.position, self.entry_price = 0.0, None

    # CRITICAL: Using the full 13-argument signature to prevent errors.
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
        """Called whenever a basketball game event occurs. Handles fundamental logic."""
        if home_away != "home": return

        event_key = event_type
        if event_type == "SHOT_MADE":
            if "3PT" in shot_type: event_key = "SHOT_MADE_3PT"
            elif "2PT" in shot_type: event_key = "SHOT_MADE_2PT"
            elif "1PT" in shot_type: event_key = "SHOT_MADE_1PT"
        if event_type == "REBOUND" and rebound_type == "OFFENSIVE":
            event_key = "REBOUND_OFFENSIVE"

        if event_key in self.event_weights:
            self.momentum_score += self.event_weights[event_key]

        if event_type == "END_GAME":
            self.reset_state()
            return

    def on_orderbook_snapshot(self, ticker: Ticker, bids: list, asks: list) -> None:
        pass