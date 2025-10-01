"""
Quant Challenge 2025

Algorithmic strategy template
"""

from enum import Enum
from typing import Optional
# Added imports required for this specific strategy
from collections import deque
import numpy as np


class Side(Enum):
    BUY = 0
    SELL = 1

class Ticker(Enum):
    # TEAM_A (home team)
    TEAM_A = 0

def place_market_order(side: Side, ticker: Ticker, quantity: float) -> None:
    """Place a market order.
    
    Parameters
    ----------
    side
        Side of order to place
    ticker
        Ticker of order to place
    quantity
        Quantity of order to place
    """
    # This is a placeholder for the actual exchange connection
    print(f"--- Placing Market Order: {side.name} {quantity} {ticker.name} ---")
    return

def place_limit_order(side: Side, ticker: Ticker, quantity: float, price: float, ioc: bool = False) -> int:
    """Place a limit order.
    
    Parameters
    ----------
    side
        Side of order to place
    ticker
        Ticker of order to place
    quantity
        Quantity of order to place
    price
        Price of order to place
    ioc
        Immediate or cancel flag (FOK)

    Returns
    -------
    order_id
        Order ID of order placed
    """
    return 0

def cancel_order(ticker: Ticker, order_id: int) -> bool:
    """Cancel an order.
    
    Parameters
    ----------
    ticker
        Ticker of order to cancel
    order_id
        Order ID of order to cancel

    Returns
    -------
    success
        True if order was cancelled, False otherwise
    """
    return True

class Strategy:
    """Template for a strategy."""

    def reset_state(self) -> None:
        """Reset the state of the strategy to the start of game position.
        
        Since the sandbox execution can start mid-game, we recommend creating a
        function which can be called from __init__ and on_game_event_update (END_GAME).

        Note: In production execution, the game will start from the beginning
        and will not be replayed.
        """
        # --- Strategy Parameters (Tunable) ---
        self.trade_size = 10.0
        self.position_limit = 50.0
        self.base_price_window = 20

        # --- Event Weights (Crucial for tuning!) ---
        self.event_weights = {
            "SHOT_MADE_3PT": 3.0, "SHOT_MADE_2PT": 2.0, "SHOT_MADE_1PT": 1.0,
            "REBOUND_OFFENSIVE": 1.5, "REBOUND_DEFENSIVE": 0.5,
            "TURNOVER": -2.0, "FOUL": -1.0,
        }

        # --- State Variables ---
        self.position = 0.0
        self.momentum_score = 0.0
        self.base_price = None
        self.last_market_price = None
        self.initial_trades = []
        
        print("Game Event Momentum Strategy state has been reset.")

    def __init__(self) -> None:
        """Your initialization code goes here."""
        self.reset_state()

    # --- Private helper method for this strategy ---
    def _update_position(self) -> None:
        """Helper to decide and execute trades based on the fair value."""
        if self.base_price is None or self.last_market_price is None:
            return

        fair_value = self.base_price + self.momentum_score
        
        if self.last_market_price < fair_value and self.position < self.position_limit:
            place_market_order(Side.BUY, Ticker.TEAM_A, self.trade_size)
        elif self.last_market_price > fair_value and self.position > -self.position_limit:
            place_market_order(Side.SELL, Ticker.TEAM_A, self.trade_size)

    def on_trade_update(
        self, ticker: Ticker, side: Side, quantity: float, price: float
    ) -> None:
        """Called whenever two orders match. Could be one of your orders, or two other people's orders.
        Parameters
        ----------
        ticker
            Ticker of orders that were matched
        side:
            Side of orders that were matched
        quantity
            Volume traded
        price
            Price that trade was executed at
        """
        self.last_market_price = price

        if self.base_price is None:
            self.initial_trades.append(price)
            if len(self.initial_trades) == self.base_price_window:
                self.base_price = np.mean(self.initial_trades)

    def on_orderbook_update(
        self, ticker: Ticker, side: Side, quantity: float, price: float
    ) -> None:
        """Called whenever the orderbook changes. This could be because of a trade, or because of a new order, or both.
        Parameters
        ----------
        ticker
            Ticker that has an orderbook update
        side
            Which orderbook was updated
        price
            Price of orderbook that has an update
        quantity
            Volume placed into orderbook
        """
        pass

    def on_account_update(
        self,
        ticker: Ticker,
        side: Side,
        price: float,
        quantity: float,
        capital_remaining: float,
    ) -> None:
        """Called whenever one of your orders is filled.
        Parameters
        ----------
        ticker
            Ticker of order that was fulfilled
        side
            Side of order that was fulfilled
        price
            Price that order was fulfilled at
        quantity
            Volume of order that was fulfilled
        capital_remaining
            Amount of capital after fulfilling order
        """
        if side == Side.BUY:
            self.position += quantity
        else:
            self.position -= quantity

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
        """Called whenever a basketball game event occurs.
        Parameters
        ----------
        event_type
            Type of event that occurred
        home_score
            Home team score after event
        away_score
            Away team score after event
        player_name (Optional)
            Player involved in event
        substituted_player_name (Optional)
            Player being substituted out
        shot_type (Optional)
            Type of shot
        assist_player (Optional)
            Player who made the assist
        rebound_type (Optional)
            Type of rebound
        coordinate_x (Optional)
            X coordinate of shot location in feet
        coordinate_y (Optional)
            Y coordinate of shot location in feet
        time_seconds (Optional)
            Game time remaining in seconds
        """
        if home_away != "home":
            return
            
        event_key = event_type
        if event_type == "SHOT_MADE":
            if "3PT" in shot_type: event_key = "SHOT_MADE_3PT"
            elif "2PT" in shot_type: event_key = "SHOT_MADE_2PT"
            elif "1PT" in shot_type: event_key = "SHOT_MADE_1PT"
        
        if event_type == "REBOUND":
            if rebound_type == "OFFENSIVE": event_key = "REBOUND_OFFENSIVE"
            elif rebound_type == "DEFENSIVE": event_key = "REBOUND_DEFENSIVE"

        if event_key in self.event_weights:
            change = self.event_weights[event_key]
            self.momentum_score += change
            self._update_position()

        if event_type == "END_GAME":
            self.reset_state()
            return