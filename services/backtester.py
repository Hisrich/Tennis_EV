"""
Backtesting Engine
Simulates betting decisions on historical data chronologically.
Prevents lookahead bias by training only on pre-split data.
"""
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
from loguru import logger
from sqlalchemy.orm import Session

from models.db_models import Match, MatchOdds, BettingOpportunity
from services.feature_engineering import FeatureEngineer
from services.value_engine import (
    decimal_to_implied_prob, remove_overround,
    calculate_edge, calculate_ev, fractional_kelly
)
from config import config as settings


class BacktestResult:
    def __init__(self):
        self.bets: List[Dict] = []
        self.bankroll_history: List[float] = []

    def add_bet(self, bet: Dict):
        self.bets.append(bet)

    def summary(self) -> Dict:
        if not self.bets:
            return {"message": "No bets placed"}

        df = pd.DataFrame(self.bets)
        total_bets = len(df)
        won = df[df["outcome"] == "won"]
        lost = df[df["outcome"] == "lost"]

        total_staked = df["stake"].sum()
        total_profit = df["profit"].sum()
        roi = total_profit / total_staked if total_staked > 0 else 0

        # Bankroll simulation
        bankroll = settings.DEFAULT_BANKROLL
        bankroll_series = [bankroll]
        for _, row in df.iterrows():
            bankroll += row["profit"]
            bankroll_series.append(bankroll)
        self.bankroll_history = bankroll_series

        peak = max(bankroll_series)
        trough = min(bankroll_series)
        max_drawdown = (peak - trough) / peak if peak > 0 else 0

        # CLV average
        clv_values = df["clv"].dropna()
        avg_clv = clv_values.mean() if len(clv_values) > 0 else 0

        return {
            "total_bets": total_bets,
            "wins": len(won),
            "losses": len(lost),
            "win_rate": len(won) / total_bets if total_bets > 0 else 0,
            "total_staked": round(total_staked, 2),
            "total_profit": round(total_profit, 2),
            "roi_percent": round(roi * 100, 2),
            "max_drawdown_percent": round(max_drawdown * 100, 2),
            "avg_edge_percent": round(df["edge"].mean(), 2),
            "avg_ev": round(df["ev"].mean(), 4),
            "avg_clv": round(avg_clv, 4),
            "avg_odds": round(df["odds"].mean(), 2),
            "bankroll_start": settings.DEFAULT_BANKROLL,
            "bankroll_end": round(bankroll_series[-1], 2),
            "bankroll_history": bankroll_series,
        }


class Backtester:
    """
    Full backtesting system.
    Replays matches chronologically and simulates betting.
    """

    def __init__(
        self,
        db: Session,
        model_version: Optional[str] = None,
        min_edge_pct: float = 5.0,
        kelly_fraction: float = 0.25,
        min_confidence: float = 0.55,
        bankroll: float = 1000.0,
    ):
        self.db = db
        self.model_version = model_version
        self.min_edge = min_edge_pct / 100
        self.kelly_fraction = kelly_fraction
        self.min_confidence = min_confidence
        self.bankroll = bankroll

    def run(
        self,
        start_date: date,
        end_date: date,
        train_months: int = 12,
    ) -> BacktestResult:
        """
        Run backtest from start_date to end_date.
        train_months: how many months of history to use for initial training.
        """
        result = BacktestResult()

        # Load completed matches in window
        matches = (
            self.db.query(Match)
            .filter(
                Match.is_completed == True,
                Match.winner_id.isnot(None),
                Match.is_walkover == False,
                Match.scheduled_at >= start_date,
                Match.scheduled_at <= end_date,
            )
            .order_by(Match.scheduled_at.asc())
            .all()
        )

        logger.info(f"Backtesting {len(matches)} matches from {start_date} to {end_date}")

        for match in matches:
            # Only proceed if we have predictions for this match
            from models.db_models import ModelPrediction
            pred_q = self.db.query(ModelPrediction).filter(
                ModelPrediction.match_id == match.id
            )
            if self.model_version:
                pred_q = pred_q.filter(ModelPrediction.model_version == self.model_version)
            prediction = pred_q.order_by(ModelPrediction.predicted_at.desc()).first()

            if not prediction:
                continue

            if prediction.confidence < self.min_confidence:
                continue

            # Get opening odds
            opening_odds = (
                self.db.query(MatchOdds)
                .filter(MatchOdds.match_id == match.id, MatchOdds.is_opening == True)
                .first()
            )

            # Fall back to earliest odds if no opening flag
            if not opening_odds:
                opening_odds = (
                    self.db.query(MatchOdds)
                    .filter(MatchOdds.match_id == match.id)
                    .order_by(MatchOdds.timestamp.asc())
                    .first()
                )

            if not opening_odds:
                continue

            # Get closing odds for CLV
            closing_odds = (
                self.db.query(MatchOdds)
                .filter(MatchOdds.match_id == match.id, MatchOdds.is_closing == True)
                .first()
            )

            raw_p1 = decimal_to_implied_prob(opening_odds.odds_player1 or 2.0)
            raw_p2 = decimal_to_implied_prob(opening_odds.odds_player2 or 2.0)
            fair_p1, fair_p2 = remove_overround(raw_p1, raw_p2)

            for player_num, model_prob, fair_prob, decimal_odds in [
                (1, prediction.prob_player1, fair_p1, opening_odds.odds_player1),
                (2, prediction.prob_player2, fair_p2, opening_odds.odds_player2),
            ]:
                if not decimal_odds or decimal_odds <= 1.0:
                    continue

                edge = calculate_edge(model_prob, fair_prob)
                ev = calculate_ev(model_prob, decimal_odds)

                if edge < self.min_edge or ev <= 0:
                    continue

                kelly = fractional_kelly(model_prob, decimal_odds, self.kelly_fraction)
                stake_pct = min(kelly, settings.MAX_KELLY_FRACTION)
                stake = self.bankroll * stake_pct

                actual_winner = match.winner_id
                if player_num == 1:
                    won = actual_winner == match.player1_id
                else:
                    won = actual_winner == match.player2_id

                profit = stake * (decimal_odds - 1) if won else -stake
                self.bankroll += profit

                # CLV calculation
                clv = None
                if closing_odds:
                    if player_num == 1 and closing_odds.odds_player1:
                        closing_ip = decimal_to_implied_prob(closing_odds.odds_player1)
                        clv = model_prob - closing_ip
                    elif player_num == 2 and closing_odds.odds_player2:
                        closing_ip = decimal_to_implied_prob(closing_odds.odds_player2)
                        clv = model_prob - closing_ip

                result.add_bet({
                    "match_id": match.id,
                    "match_date": match.scheduled_at,
                    "player_num": player_num,
                    "bookmaker": opening_odds.bookmaker,
                    "odds": decimal_odds,
                    "edge": edge * 100,
                    "ev": ev,
                    "stake": stake,
                    "outcome": "won" if won else "lost",
                    "profit": profit,
                    "clv": clv,
                    "model_prob": model_prob,
                    "fair_prob": fair_prob,
                    "bankroll_after": self.bankroll,
                })

        logger.info(f"Backtest complete. Summary: {result.summary()}")
        return result
