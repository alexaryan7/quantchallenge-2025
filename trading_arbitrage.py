import random
from enum import Enum
from typing import Optional
from collections import deque
import numpy as np

def place_market_order(side: 'Side', ticker: 'Ticker', quantity: float) -> None:
    print(f"--- Placing Market Order: {side.name} {quantity} {ticker.name} ---")
    return

def place_limit_order(side: 'Side', ticker: 'Ticker', quantity: float, price: float, ioc: bool = False) -> int:
    order_id = random.randint(1000, 9999)
    return order_id

def cancel_order(ticker: 'Ticker', order_id: int) -> bool:
    return True

class Side(Enum):
    BUY = 0
    SELL = 1

class Ticker(Enum):
    TEAM_A = 0

class Strategy:
    def reset_state(self) -> None:
        self.trade_size = 10.0
        self.position_limit = 50.0
        self.base_price_window = 20

        self.event_weights = {
            "SHOT_MADE_3PT": 3.0,
            "SHOT_MADE_2PT": 2.0,
            "SHOT_MADE_1PT": 1.0,
            "REBOUND_OFFENSIVE": 1.5,
            "REBOUND_DEFENSIVE": 0.5,
            "TURNOVER": -2.0,
            "FOUL": -1.0,
        }

        self.position = 0.0
        self.momentum_score = 0.0
        self.base_price = None
        self.last_market_price = None
        self.initial_trades = []
        
        print("Game Event Momentum Strategy state has been reset.")

    def __init__(self) -> None:
        self.reset_state()

    def _update_position(self) -> None:
        if self.base_price is None or self.last_market_price is None:
            return

        fair_value = self.base_price + self.momentum_score
        
        if self.last_market_price < fair_value and self.position < self.position_limit:
            print(f"Signal: Market Price ({self.last_market_price:.2f}) < Fair Value ({fair_value:.2f}). Buying.")
            place_market_order(Side.BUY, Ticker.TEAM_A, self.trade_size)

        elif self.last_market_price > fair_value and self.position > -self.position_limit:
            print(f"Signal: Market Price ({self.last_market_price:.2f}) > Fair Value ({fair_value:.2f}). Selling.")
            place_market_order(Side.SELL, Ticker.TEAM_A, self.trade_size)
            
    def on_trade_update(
        self, ticker: Ticker, side: Side, quantity: float, price: float
    ) -> None:
        self.last_market_price = price

        if self.base_price is None:
            self.initial_trades.append(price)
            if len(self.initial_trades) == self.base_price_window:
                self.base_price = np.mean(self.initial_trades)
                print(f"Base price established at: {self.base_price:.2f}")

    def on_account_update(
        self, ticker: Ticker, side: Side, price: float, quantity: float, capital_remaining: float,
    ) -> None:
        if side == Side.BUY:
            self.position += quantity
        else:
            self.position -= quantity
        print(f"ACCOUNT UPDATE: Position is now {self.position}")

    def on_game_event_update(
        self, event_type: str, home_away: str, **kwargs
    ) -> None:
        if home_away != "home":
            return
            
        event_key = event_type
        shot_type = kwargs.get("shot_type")
        if event_type == "SHOT_MADE":
            if "3PT" in shot_type: event_key = "SHOT_MADE_3PT"
            elif "2PT" in shot_type: event_key = "SHOT_MADE_2PT"
            elif "1PT" in shot_type: event_key = "SHOT_MADE_1PT"
        
        rebound_type = kwargs.get("rebound_type")
        if event_type == "REBOUND":
            if rebound_type == "OFFENSIVE": event_key = "REBOUND_OFFENSIVE"
            elif rebound_type == "DEFENSIVE": event_key = "REBOUND_DEFENSIVE"

        if event_key in self.event_weights:
            change = self.event_weights[event_key]
            self.momentum_score += change
            print(f"Game Event: '{event_key}' for TEAM_A. Momentum change: {change:+.2f}. New Score: {self.momentum_score:.2f}")
            self._update_position()
        
        if event_type == "END_GAME":
            print("Game has ended. Resetting strategy.")
            self.reset_state()

    def on_orderbook_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        pass