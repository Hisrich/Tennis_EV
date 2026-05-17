"""
Feature Engineering Pipeline
Generates all features for ML model training and inference.
All features use only pre-match information to prevent data leakage.
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
from loguru import logger

from models.db_models import Match, Player, EloRating, MatchOdds, PlayerStats
from services.elo_engine import EloEngine


class FeatureEngineer:
    """
    Generates feature vectors for match prediction.
    All lookups are scoped to data BEFORE the match date.
    """

    def __init__(self, db: Session):
        self.db = db
        self.elo = EloEngine(db)

    def build_features(self, match: Match, reference_date: Optional[datetime] = None) -> Optional[Dict]:
        """
        Build full feature dict for a single match.
        reference_date: use match.scheduled_at if None (for inference), or a past date (for backtesting).
        """
        ref_dt = reference_date or match.scheduled_at
        if ref_dt is None:
            return None

        ref_date = ref_dt.date() if isinstance(ref_dt, datetime) else ref_dt

        p1 = match.player1
        p2 = match.player2
        if not p1 or not p2:
            return None

        surface = match.surface if isinstance(match.surface, str) else (match.surface.value if match.surface else None)
        features = {}

        # ── 1. Elo Features ──────────────────────────────────────────
        r1_overall = self._get_elo_at(p1.id, ref_date)
        r2_overall = self._get_elo_at(p2.id, ref_date)
        features["elo_diff"] = r1_overall - r2_overall
        features["elo_p1"] = r1_overall
        features["elo_p2"] = r2_overall

        if surface:
            r1_surf = self._get_elo_at(p1.id, ref_date, surface)
            r2_surf = self._get_elo_at(p2.id, ref_date, surface)
            features["elo_surface_diff"] = r1_surf - r2_surf
            features["elo_surface_p1"] = r1_surf
            features["elo_surface_p2"] = r2_surf
        else:
            features["elo_surface_diff"] = 0
            features["elo_surface_p1"] = r1_overall
            features["elo_surface_p2"] = r2_overall

        # Elo momentum: change over last 30 days
        features["elo_momentum_p1"] = self._elo_momentum(p1.id, ref_date)
        features["elo_momentum_p2"] = self._elo_momentum(p2.id, ref_date)

        # ── 2. Ranking Features ────────────────────────────────────────
        rank1 = match.player1_rank or p1.ranking or 999
        rank2 = match.player2_rank or p2.ranking or 999
        features["rank_p1"] = rank1
        features["rank_p2"] = rank2
        features["rank_diff"] = rank1 - rank2
        features["log_rank_ratio"] = np.log((rank2 + 1) / (rank1 + 1))

        # ── 3. Form Features ──────────────────────────────────────────
        for window in [5, 10, 20]:
            w1 = self._win_rate(p1.id, ref_date, window)
            w2 = self._win_rate(p2.id, ref_date, window)
            features[f"win_rate_{window}_p1"] = w1
            features[f"win_rate_{window}_p2"] = w2
            features[f"win_rate_{window}_diff"] = w1 - w2

        if surface:
            for window in [10, 20]:
                w1s = self._win_rate(p1.id, ref_date, window, surface=surface)
                w2s = self._win_rate(p2.id, ref_date, window, surface=surface)
                features[f"win_rate_surface_{window}_p1"] = w1s
                features[f"win_rate_surface_{window}_p2"] = w2s
                features[f"win_rate_surface_{window}_diff"] = w1s - w2s

        # ── 4. Fatigue Features ─────────────────────────────────────
        features["matches_7d_p1"] = self._matches_in_days(p1.id, ref_date, 7)
        features["matches_7d_p2"] = self._matches_in_days(p2.id, ref_date, 7)
        features["matches_14d_p1"] = self._matches_in_days(p1.id, ref_date, 14)
        features["matches_14d_p2"] = self._matches_in_days(p2.id, ref_date, 14)
        features["days_rest_p1"] = self._days_since_last_match(p1.id, ref_date)
        features["days_rest_p2"] = self._days_since_last_match(p2.id, ref_date)
        features["fatigue_diff"] = features["matches_7d_p1"] - features["matches_7d_p2"]

        # ── 5. H2H Features ─────────────────────────────────────────
        h2h = self._head_to_head(p1.id, p2.id, ref_date)
        features["h2h_total"] = h2h["total"]
        features["h2h_p1_wins"] = h2h["p1_wins"]
        features["h2h_p2_wins"] = h2h["p2_wins"]
        features["h2h_p1_win_rate"] = h2h["p1_win_rate"]

        if surface:
            h2h_surf = self._head_to_head(p1.id, p2.id, ref_date, surface=surface)
            features["h2h_surface_total"] = h2h_surf["total"]
            features["h2h_surface_p1_win_rate"] = h2h_surf["p1_win_rate"]
        else:
            features["h2h_surface_total"] = 0
            features["h2h_surface_p1_win_rate"] = 0.5

        # ── 6. Player Profile Features ───────────────────────────────
        features["age_p1"] = p1.age or 25.0
        features["age_p2"] = p2.age or 25.0
        features["age_diff"] = features["age_p1"] - features["age_p2"]
        features["height_diff"] = (p1.height_cm or 183) - (p2.height_cm or 183)
        features["p1_right_handed"] = 1 if (p1.handedness or "right").lower() == "right" else 0
        features["p2_right_handed"] = 1 if (p2.handedness or "right").lower() == "right" else 0

        # ── 7. Surface Encoding ──────────────────────────────────────
        surfaces = ["hard", "clay", "grass", "carpet"]
        for s in surfaces:
            features[f"surface_{s}"] = 1 if surface == s else 0

        # ── 8. Tournament Context ────────────────────────────────────
        if match.tournament:
            features["is_grand_slam"] = 1 if "Grand Slam" in (match.tournament.category or "") else 0
            features["is_masters"] = 1 if "Masters" in (match.tournament.category or "") else 0
            features["draw_size"] = match.tournament.draw_size or 32
        else:
            features["is_grand_slam"] = 0
            features["is_masters"] = 0
            features["draw_size"] = 32

        round_map = {"R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7}
        features["round_num"] = round_map.get(match.round or "", 3)

        # ── 9. Market Features (if odds available) ───────────────────
        market = self._get_market_features(match.id, ref_dt)
        features.update(market)

        # ── 10. Data quality check ───────────────────────────────────
        # If both players have default Elo and unknown rankings,
        # the prediction will be meaningless — skip it.
        elo_p1 = features.get("elo_p1", 1500.0)
        elo_p2 = features.get("elo_p2", 1500.0)
        rank_p1 = features.get("rank_p1", 999)
        rank_p2 = features.get("rank_p2", 999)

        if elo_p1 == 1500.0 and elo_p2 == 1500.0 and rank_p1 == 999 and rank_p2 == 999:
            logger.warning(
                f"Match {match.id} ({p1.name} vs {p2.name}): "
                f"both players have no Elo history and unknown rankings, skipping"
            )
            return None

        return features

    def build_dataset(self, matches: List[Match]) -> pd.DataFrame:
        """Build feature matrix for a list of completed matches."""
        rows = []
        for match in matches:
            if not match.is_completed or match.winner_id is None:
                continue
            feats = self.build_features(match, reference_date=match.scheduled_at)
            if feats is None:
                continue
            # Remove market features from training to prevent circular reasoning:
            # the model would learn to agree with the market, then find "value"
            # by comparing its own market-influenced prediction back to the market.
            feats.pop("market_prob_p1", None)
            feats.pop("odds_movement_p1", None)
            feats.pop("bookmaker_count", None)
            feats.pop("has_market_data", None)
            feats["target"] = 1 if match.winner_id == match.player1_id else 0
            feats["match_id"] = match.id
            feats["match_date"] = match.scheduled_at
            rows.append(feats)

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    # ─────────────────── Private Helpers ───────────────────────────

    def _get_elo_at(self, player_id: int, ref_date, surface: Optional[str] = None) -> float:
        record = (
            self.db.query(EloRating)
            .filter(
                EloRating.player_id == player_id,
                EloRating.surface == surface,
                EloRating.date < ref_date,
            )
            .order_by(EloRating.date.desc())
            .first()
        )
        return record.rating if record else 1500.0

    def _elo_momentum(self, player_id: int, ref_date, days: int = 30) -> float:
        past = ref_date - timedelta(days=days)
        current = self._get_elo_at(player_id, ref_date)
        previous = self._get_elo_at(player_id, past)
        return current - previous

    def _win_rate(
        self, player_id: int, ref_date, window: int, surface: Optional[str] = None
    ) -> float:
        cutoff = ref_date - timedelta(days=730)  # Don't go too far back
        q = (
            self.db.query(Match)
            .filter(
                or_(Match.player1_id == player_id, Match.player2_id == player_id),
                Match.is_completed == True,
                Match.winner_id.isnot(None),
                Match.scheduled_at < ref_date,
                Match.scheduled_at >= cutoff,
            )
        )
        if surface:
            q = q.filter(Match.surface == surface)

        recent = q.order_by(Match.scheduled_at.desc()).limit(window).all()
        if not recent:
            return 0.5
        wins = sum(1 for m in recent if m.winner_id == player_id)
        return wins / len(recent)

    def _matches_in_days(self, player_id: int, ref_date, days: int) -> int:
        cutoff = ref_date - timedelta(days=days)
        return (
            self.db.query(func.count(Match.id))
            .filter(
                or_(Match.player1_id == player_id, Match.player2_id == player_id),
                Match.is_completed == True,
                Match.scheduled_at >= cutoff,
                Match.scheduled_at < ref_date,
            )
            .scalar()
        )

    def _days_since_last_match(self, player_id: int, ref_date) -> float:
        last = (
            self.db.query(Match)
            .filter(
                or_(Match.player1_id == player_id, Match.player2_id == player_id),
                Match.is_completed == True,
                Match.scheduled_at < ref_date,
            )
            .order_by(Match.scheduled_at.desc())
            .first()
        )
        if not last or not last.scheduled_at:
            return 14.0  # Default: assume 2 weeks rest
        delta = ref_date - last.scheduled_at.date()
        return float(delta.days)

    def _head_to_head(
        self, p1_id: int, p2_id: int, ref_date, surface: Optional[str] = None
    ) -> Dict:
        q = (
            self.db.query(Match)
            .filter(
                or_(
                    and_(Match.player1_id == p1_id, Match.player2_id == p2_id),
                    and_(Match.player1_id == p2_id, Match.player2_id == p1_id),
                ),
                Match.is_completed == True,
                Match.winner_id.isnot(None),
                Match.scheduled_at < ref_date,
            )
        )
        if surface:
            q = q.filter(Match.surface == surface)

        h2h_matches = q.all()
        total = len(h2h_matches)
        if total == 0:
            return {"total": 0, "p1_wins": 0, "p2_wins": 0, "p1_win_rate": 0.5}

        p1_wins = sum(1 for m in h2h_matches if m.winner_id == p1_id)
        return {
            "total": total,
            "p1_wins": p1_wins,
            "p2_wins": total - p1_wins,
            "p1_win_rate": p1_wins / total,
        }

    def _get_market_features(self, match_id: int, ref_dt: datetime) -> Dict:
        """Extract consensus market probability and line movement."""
        odds_records = (
            self.db.query(MatchOdds)
            .filter(MatchOdds.match_id == match_id, MatchOdds.timestamp <= ref_dt)
            .all()
        )
        if not odds_records:
            return {
                "market_prob_p1": 0.5,
                "odds_movement_p1": 0.0,
                "bookmaker_count": 0,
                "has_market_data": 0,
            }

        # Average implied probability across bookmakers
        avg_ip1 = np.mean([o.implied_prob_p1 for o in odds_records if o.implied_prob_p1])

        # Line movement: opening vs latest
        opening = next((o for o in odds_records if o.is_opening), None)
        latest = odds_records[-1]
        movement = 0.0
        if opening and opening.implied_prob_p1 and latest.implied_prob_p1:
            movement = latest.implied_prob_p1 - opening.implied_prob_p1

        return {
            "market_prob_p1": float(avg_ip1) if avg_ip1 else 0.5,
            "odds_movement_p1": movement,
            "bookmaker_count": len(set(o.bookmaker for o in odds_records)),
            "has_market_data": 1,
        }
