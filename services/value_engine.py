"""
Value Betting Engine
The core of the platform.
Identifies mispriced bookmaker probabilities using:
  - ML model probability
  - Bookmaker implied probability
  - Edge calculation
  - Kelly Criterion staking
"""
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from loguru import logger
from sqlalchemy.orm import Session

from models.db_models import (
    Match, MatchOdds, ModelPrediction, BettingOpportunity
)
from config import config as settings


def decimal_to_implied_prob(odds: float, overround_adjust: bool = True) -> float:
    """
    Convert decimal odds to implied probability.
    Raw: 1/odds. Overround adjustment requires all lines.
    """
    if odds <= 1.0:
        return 1.0
    return 1.0 / odds


def remove_overround(implied_p1: float, implied_p2: float) -> Tuple[float, float]:
    """
    Remove bookmaker margin (overround) to get true implied probabilities.
    Sum of raw implied probs > 1.0 by the margin.
    """
    total = implied_p1 + implied_p2
    if total <= 0:
        return 0.5, 0.5
    return implied_p1 / total, implied_p2 / total


def calculate_edge(model_prob: float, fair_implied_prob: float) -> float:
    """
    Edge = model probability - fair implied probability
    Positive edge = value bet
    """
    return model_prob - fair_implied_prob


def calculate_ev(model_prob: float, decimal_odds: float) -> float:
    """
    Expected Value = (model_prob × net_win) - (1 - model_prob)
    For a 1-unit bet at decimal odds D:
      EV = model_prob × (D - 1) - (1 - model_prob)
    """
    return model_prob * (decimal_odds - 1) - (1 - model_prob)


def kelly_criterion(model_prob: float, decimal_odds: float) -> float:
    """
    Full Kelly fraction:
      f* = (b*p - q) / b
    where b = decimal_odds - 1, p = model_prob, q = 1 - p
    """
    b = decimal_odds - 1
    p = model_prob
    q = 1.0 - p
    if b <= 0:
        return 0.0
    f = (b * p - q) / b
    return max(0.0, f)


def fractional_kelly(model_prob: float, decimal_odds: float, fraction: float = 0.25) -> float:
    """Fractional Kelly — safer for real-world use."""
    return kelly_criterion(model_prob, decimal_odds) * fraction


class ValueBettingEngine:
    """
    Scans upcoming matches with predictions and available odds
    to find positive EV opportunities.
    """

    def __init__(self, db: Session):
        self.db = db
        self.min_edge = settings.MIN_EDGE_PERCENT / 100.0
        self.min_confidence = settings.MIN_CONFIDENCE
        self.kelly_fraction = settings.KELLY_FRACTION
        self.max_kelly = settings.MAX_KELLY_FRACTION

    def scan_match(self, match: Match) -> List[BettingOpportunity]:
        """
        Scan a single match for value betting opportunities.
        Returns a list of BettingOpportunity objects (not yet saved).
        """
        opportunities = []

        # Get the latest prediction
        prediction = (
            self.db.query(ModelPrediction)
            .filter(ModelPrediction.match_id == match.id)
            .order_by(ModelPrediction.predicted_at.desc())
            .first()
        )
        if not prediction:
            return []

        if prediction.confidence < self.min_confidence:
            logger.debug(f"Match {match.id}: confidence {prediction.confidence:.2f} below threshold")
            return []

        # Get latest odds per bookmaker
        odds_records = (
            self.db.query(MatchOdds)
            .filter(MatchOdds.match_id == match.id)
            .order_by(MatchOdds.timestamp.desc())
            .all()
        )

        if not odds_records:
            return []

        # Group by bookmaker: take the latest record for each
        latest_by_bookie: Dict[str, MatchOdds] = {}
        for record in odds_records:
            if record.bookmaker not in latest_by_bookie:
                latest_by_bookie[record.bookmaker] = record

        for bookie, odds in latest_by_bookie.items():
            opps = self._evaluate_odds(match, prediction, odds)
            opportunities.extend(opps)

        return opportunities

    def _evaluate_odds(
        self,
        match: Match,
        prediction: ModelPrediction,
        odds: MatchOdds,
    ) -> List[BettingOpportunity]:
        """Check both sides of the market for value."""
        found = []

        raw_p1 = decimal_to_implied_prob(odds.odds_player1)
        raw_p2 = decimal_to_implied_prob(odds.odds_player2)
        fair_p1, fair_p2 = remove_overround(raw_p1, raw_p2)

        model_p1 = prediction.prob_player1
        model_p2 = prediction.prob_player2

        for player_num, model_prob, fair_prob, decimal_odds in [
            (1, model_p1, fair_p1, odds.odds_player1),
            (2, model_p2, fair_p2, odds.odds_player2),
        ]:
            if not decimal_odds or decimal_odds <= 1.0:
                continue

            edge = calculate_edge(model_prob, fair_prob)
            ev = calculate_ev(model_prob, decimal_odds)
            kelly = fractional_kelly(model_prob, decimal_odds, self.kelly_fraction)
            rec_stake = min(kelly, self.max_kelly)

            if edge >= self.min_edge and ev > 0:
                logger.info(
                    f"VALUE BET: Match {match.id} | Player {player_num} | "
                    f"Odds {decimal_odds} | Edge {edge*100:.1f}% | EV {ev:.3f} | "
                    f"Stake {rec_stake*100:.1f}%"
                )
                opp = BettingOpportunity(
                    match_id=match.id,
                    prediction_id=prediction.id,
                    bet_on_player=player_num,
                    bookmaker=odds.bookmaker,
                    decimal_odds=decimal_odds,
                    implied_probability=fair_prob,
                    model_probability=model_prob,
                    edge_percent=edge * 100,
                    expected_value=ev,
                    kelly_stake_pct=kelly * 100,
                    recommended_stake_pct=rec_stake * 100,
                )
                found.append(opp)

        return found

    def scan_all_upcoming(self) -> List[BettingOpportunity]:
        """
        Scan all upcoming (not completed) matches.
        Saves new opportunities and returns them.
        """
        from datetime import datetime, timezone
        now = datetime.utcnow()

        upcoming = (
            self.db.query(Match)
            .filter(
                Match.is_completed == False,
                Match.scheduled_at > now,
            )
            .all()
        )

        logger.info(f"Scanning {len(upcoming)} upcoming matches for value...")
        all_opps = []

        for match in upcoming:
            opps = self.scan_match(match)
            for opp in opps:
                # Check not already detected
                existing = (
                    self.db.query(BettingOpportunity)
                    .filter(
                        BettingOpportunity.match_id == opp.match_id,
                        BettingOpportunity.bet_on_player == opp.bet_on_player,
                        BettingOpportunity.bookmaker == opp.bookmaker,
                    )
                    .first()
                )
                if not existing:
                    self.db.add(opp)
                    all_opps.append(opp)

        self.db.commit()
        logger.info(f"Found {len(all_opps)} new value betting opportunities")
        return all_opps
