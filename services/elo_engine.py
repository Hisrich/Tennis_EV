"""
Elo Rating Engine
- Dynamic K-factor
- Surface-specific ratings
- Historical recalculation support
"""
import math
from datetime import date, datetime
from typing import Optional, Dict, Tuple
from loguru import logger
from sqlalchemy.orm import Session

from models.db_models import Player, Match, EloRating, Surface


# ─────────────────────── Constants ───────────────────────

DEFAULT_ELO = 1500.0
SURFACE_WEIGHT = 0.6        # How much surface-Elo blends with overall Elo
MIN_MATCHES_FAST_K = 10     # K ramps down after this many matches


def get_k_factor(matches_played: int, surface_matches: int = 0) -> float:
    """Dynamic K-factor: higher for new players, stabilizes over time."""
    if matches_played < 5:
        return 40.0
    elif matches_played < 10:
        return 32.0
    elif matches_played < 30:
        return 24.0
    else:
        return 20.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected win probability for player A against player B."""
    return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))


def update_elo(
    rating_a: float,
    rating_b: float,
    score_a: float,   # 1.0 if A won, 0.0 if B won
    k_factor: float = 24.0,
) -> Tuple[float, float]:
    """Return updated (rating_a, rating_b)."""
    exp_a = expected_score(rating_a, rating_b)
    exp_b = 1.0 - exp_a
    new_a = rating_a + k_factor * (score_a - exp_a)
    new_b = rating_b + k_factor * ((1.0 - score_a) - exp_b)
    return new_a, new_b


# ─────────────────────── Main Engine ───────────────────────

class EloEngine:
    """
    Manages overall and surface-specific Elo ratings.
    All updates are idempotent — safe to re-run.
    """

    def __init__(self, db: Session):
        self.db = db
        # In-memory cache: {(player_id, surface_or_None): rating}
        self._cache: Dict[Tuple[int, Optional[str]], float] = {}
        self._matches_cache: Dict[Tuple[int, Optional[str]], int] = {}

    def _cache_key(self, player_id: int, surface: Optional[str]) -> Tuple:
        return (player_id, surface)

    def get_rating(self, player_id: int, surface: Optional[str] = None) -> float:
        key = self._cache_key(player_id, surface)
        if key in self._cache:
            return self._cache[key]

        record = (
            self.db.query(EloRating)
            .filter(
                EloRating.player_id == player_id,
                EloRating.surface == surface,
            )
            .order_by(EloRating.date.desc())
            .first()
        )
        rating = record.rating if record else DEFAULT_ELO
        matches = record.matches_played if record else 0
        self._cache[key] = rating
        self._matches_cache[key] = matches
        return rating

    def get_matches_played(self, player_id: int, surface: Optional[str] = None) -> int:
        key = self._cache_key(player_id, surface)
        if key in self._matches_cache:
            return self._matches_cache[key]
        self.get_rating(player_id, surface)  # loads into cache
        return self._matches_cache.get(key, 0)

    def process_match(self, match: Match, save: bool = True) -> Dict:
        """
        Process a completed match and update Elo ratings.
        Returns dict with old/new ratings for both players.
        """
        if not match.is_completed or match.winner_id is None:
            return {}

        p1_id = match.player1_id
        p2_id = match.player2_id
        surface = match.surface if match.surface else None

        score_p1_overall = 1.0 if match.winner_id == p1_id else 0.0
        score_p1_surface = score_p1_overall

        # — Overall Elo —
        r1_overall = self.get_rating(p1_id)
        r2_overall = self.get_rating(p2_id)
        k1 = get_k_factor(self.get_matches_played(p1_id))
        k2 = get_k_factor(self.get_matches_played(p2_id))
        k_overall = (k1 + k2) / 2

        new_r1_overall, new_r2_overall = update_elo(r1_overall, r2_overall, score_p1_overall, k_overall)

        self._cache[self._cache_key(p1_id, None)] = new_r1_overall
        self._cache[self._cache_key(p2_id, None)] = new_r2_overall
        self._matches_cache[self._cache_key(p1_id, None)] = self.get_matches_played(p1_id) + 1
        self._matches_cache[self._cache_key(p2_id, None)] = self.get_matches_played(p2_id) + 1

        # — Surface Elo —
        result = {
            "p1_old_overall": r1_overall,
            "p2_old_overall": r2_overall,
            "p1_new_overall": new_r1_overall,
            "p2_new_overall": new_r2_overall,
        }

        if surface:
            r1_surf = self.get_rating(p1_id, surface)
            r2_surf = self.get_rating(p2_id, surface)
            k_surf = (
                get_k_factor(self.get_matches_played(p1_id, surface))
                + get_k_factor(self.get_matches_played(p2_id, surface))
            ) / 2

            new_r1_surf, new_r2_surf = update_elo(r1_surf, r2_surf, score_p1_surface, k_surf)

            self._cache[self._cache_key(p1_id, surface)] = new_r1_surf
            self._cache[self._cache_key(p2_id, surface)] = new_r2_surf
            self._matches_cache[self._cache_key(p1_id, surface)] = self.get_matches_played(p1_id, surface) + 1
            self._matches_cache[self._cache_key(p2_id, surface)] = self.get_matches_played(p2_id, surface) + 1

            result.update({
                "p1_old_surface": r1_surf,
                "p2_old_surface": r2_surf,
                "p1_new_surface": new_r1_surf,
                "p2_new_surface": new_r2_surf,
            })

        if save and match.completed_at:
            match_date = match.completed_at.date() if isinstance(match.completed_at, datetime) else match.completed_at
            self._save_ratings(p1_id, p2_id, surface, match_date, result)

        return result

    def _save_ratings(self, p1_id, p2_id, surface, match_date, result):
        for player_id, new_overall, new_surf in [
            (p1_id, result["p1_new_overall"], result.get("p1_new_surface")),
            (p2_id, result["p2_new_overall"], result.get("p2_new_surface")),
        ]:
            # Save/update overall
            self._upsert_elo(player_id, match_date, None, new_overall, self.get_matches_played(player_id))

            # Save/update surface
            if surface and new_surf is not None:
                self._upsert_elo(player_id, match_date, surface, new_surf, self.get_matches_played(player_id, surface))

        self.db.commit()

    def _upsert_elo(self, player_id, match_date, surface, rating, matches):
        record = (
            self.db.query(EloRating)
            .filter(
                EloRating.player_id == player_id,
                EloRating.date == match_date,
                EloRating.surface == surface,
            )
            .first()
        )
        if record:
            record.rating = rating
            record.matches_played = matches
        else:
            self.db.add(EloRating(
                player_id=player_id,
                date=match_date,
                surface=surface,
                rating=rating,
                matches_played=matches,
            ))

    def rebuild_all(self):
        """
        Recalculate all Elo ratings from scratch, chronologically.
        Useful for reseeding after schema changes.
        """
        logger.info("Starting full Elo rebuild...")

        # Clear existing
        self.db.query(EloRating).delete()
        self.db.commit()
        self._cache.clear()
        self._matches_cache.clear()

        # Load completed matches in chronological order
        matches = (
            self.db.query(Match)
            .filter(Match.is_completed == True, Match.winner_id.isnot(None))
            .order_by(Match.completed_at.asc())
            .all()
        )

        logger.info(f"Processing {len(matches)} matches for Elo rebuild...")
        for i, match in enumerate(matches):
            self.process_match(match, save=True)
            if i % 500 == 0:
                logger.info(f"  Processed {i}/{len(matches)} matches")

        logger.info("Elo rebuild complete.")

    def get_win_probability(
        self, p1_id: int, p2_id: int, surface: Optional[str] = None
    ) -> Tuple[float, float]:
        """
        Return (prob_p1_wins, prob_p2_wins) using blended overall+surface Elo.
        """
        r1_overall = self.get_rating(p1_id)
        r2_overall = self.get_rating(p2_id)

        if surface:
            r1_surf = self.get_rating(p1_id, surface)
            r2_surf = self.get_rating(p2_id, surface)
            # Blend overall and surface ratings
            r1 = (1 - SURFACE_WEIGHT) * r1_overall + SURFACE_WEIGHT * r1_surf
            r2 = (1 - SURFACE_WEIGHT) * r2_overall + SURFACE_WEIGHT * r2_surf
        else:
            r1, r2 = r1_overall, r2_overall

        prob_p1 = expected_score(r1, r2)
        return prob_p1, 1.0 - prob_p1
