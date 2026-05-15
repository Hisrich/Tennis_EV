"""
Data Collection Engine
- Fetches live odds from The Odds API
- Fetches tennis match data from open sources
- Implements retry, rate limiting, deduplication
"""
import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from sqlalchemy.orm import Session

from config import config as settings
from models.db_models import (
    Match, Player, Tournament, MatchOdds, TourLevel, Surface
)


# ─────────────────────────── Odds API ───────────────────────────

class OddsAPIClient:
    """
    Fetches betting odds from The Odds API (https://the-odds-api.com).
    Free tier: 500 requests/month.
    """

    SPORT_KEY = "tennis_atp"
    SPORTS = [
        "tennis_atp"
    ]

    def __init__(self):
        self.api_key = settings.ODDS_API_KEY
        self.base_url = settings.ODDS_API_BASE_URL
        self._last_request = 0
        self.min_interval = 1.0  # Rate limit: 1 req/sec

    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request = time.time()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_odds(self, sport: str = "tennis_atp") -> List[Dict]:
        """Fetch current odds for a sport."""
        if not self.api_key:
            logger.warning("No ODDS_API_KEY configured — using mock data")
            return self._mock_odds()

        self._rate_limit()
        url = f"{self.base_url}/sports/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "eu,uk",
            "markets": "h2h",
            "oddsFormat": "decimal",
        }
        with httpx.Client(timeout=15) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            remaining = response.headers.get("x-requests-remaining", "?")
            logger.info(f"Odds API requests remaining: {remaining}")
            return response.json()

    def _mock_odds(self) -> List[Dict]:
        """Return sample data when no API key is set."""
        return [
            {
                "id": "mock_match_001",
                "sport_key": "tennis_atp",
                "sport_title": "ATP",
                "commence_time": (datetime.utcnow() + timedelta(hours=3)).isoformat() + "Z",
                "home_team": "Novak Djokovic",
                "away_team": "Carlos Alcaraz",
                "bookmakers": [
                    {
                        "key": "bet365",
                        "title": "Bet365",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Novak Djokovic", "price": 2.20},
                                    {"name": "Carlos Alcaraz", "price": 1.70},
                                ],
                            }
                        ],
                    },
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Novak Djokovic", "price": 2.25},
                                    {"name": "Carlos Alcaraz", "price": 1.68},
                                ],
                            }
                        ],
                    },
                ],
            }
        ]


# ─────────────────────────── Ingestion ───────────────────────────

class DataIngestionService:
    """
    Processes raw API responses and writes to database.
    """

    def __init__(self, db: Session):
        self.db = db
        self.odds_client = OddsAPIClient()

    def ingest_odds(self) -> int:
        """Fetch odds for all tennis markets and store them. Returns count of new records."""
        count = 0
        for sport in OddsAPIClient.SPORTS:
            try:
                raw = self.odds_client.get_odds(sport)
                for event in raw:
                    saved = self._process_event(event)
                    count += saved
            except Exception as e:
                logger.error(f"Error fetching odds for {sport}: {e!r}")

        logger.info(f"Ingested {count} new odds records")
        return count

    def _process_event(self, event: Dict) -> int:
        """Process a single event from the odds API. Returns number of records saved."""
        home_name = event.get("home_team", "")
        away_name = event.get("away_team", "")
        commence_time_str = event.get("commence_time", "")
        external_id = event.get("id", "")

        try:
            scheduled_at = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            scheduled_at = datetime.utcnow() + timedelta(hours=2)

        # Get or create players
        player1 = self._get_or_create_player(home_name)
        player2 = self._get_or_create_player(away_name)

        # Get or create match
        match = self._get_or_create_match(external_id, player1, player2, scheduled_at, event)

        count = 0
        for bookmaker in event.get("bookmakers", []):
            bookie_name = bookmaker.get("title", bookmaker.get("key", "unknown"))
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes = market.get("outcomes", [])
                if len(outcomes) < 2:
                    continue

                # Map outcome to player
                p1_outcome = next((o for o in outcomes if o["name"] == home_name), None)
                p2_outcome = next((o for o in outcomes if o["name"] == away_name), None)

                if not p1_outcome or not p2_outcome:
                    continue

                odds_p1 = float(p1_outcome["price"])
                odds_p2 = float(p2_outcome["price"])
                raw_ip1 = 1.0 / odds_p1
                raw_ip2 = 1.0 / odds_p2
                total = raw_ip1 + raw_ip2
                fair_ip1 = raw_ip1 / total
                fair_ip2 = raw_ip2 / total

                record = MatchOdds(
                    match_id=match.id,
                    bookmaker=bookie_name,
                    market="h2h",
                    odds_player1=odds_p1,
                    odds_player2=odds_p2,
                    implied_prob_p1=fair_ip1,
                    implied_prob_p2=fair_ip2,
                    timestamp=datetime.utcnow(),
                )
                self.db.add(record)
                count += 1

        self.db.commit()
        return count

    def _get_or_create_player(self, name: str) -> Player:
        player = self.db.query(Player).filter(Player.name == name).first()
        if not player:
            player = Player(name=name, is_active=True)
            self.db.add(player)
            self.db.commit()
            self.db.refresh(player)
        return player

    def _get_or_create_match(
        self, external_id: str, player1: Player, player2: Player,
        scheduled_at: datetime, event: Dict
    ) -> Match:
        match = None
        if external_id:
            match = self.db.query(Match).filter(Match.external_id == external_id).first()

        if not match:
            match = Match(
                external_id=external_id or f"{player1.id}_{player2.id}_{scheduled_at.date()}",
                player1_id=player1.id,
                player2_id=player2.id,
                scheduled_at=scheduled_at,
                is_completed=False,
            )
            self.db.add(match)
            self.db.commit()
            self.db.refresh(match)

        return match

    def ingest_historical_from_csv(self, filepath: str) -> int:
        """
        Load historical tennis match data from CSV (e.g. Jeff Sackmann tennis_atp dataset).
        CSV should have columns: tourney_date, winner_name, loser_name, surface, round, etc.
        """
        import pandas as pd
        import pathlib
        path = pathlib.Path(filepath)
        if path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path, low_memory=False)
        df = df.dropna(subset=["winner_name", "loser_name"])

        count = 0
        for _, row in df.iterrows():
            try:
                self._ingest_historical_row(row)
                count += 1
            except Exception as e:
                logger.debug(f"Skip row: {e}")

        self.db.commit()
        logger.info(f"Ingested {count} historical matches from {filepath}")
        return count

    def _ingest_historical_row(self, row):
        """Ingest a single historical CSV row (Jeff Sackmann format)."""
        import pandas as pd
        winner_name = str(row.get("winner_name", "")).strip()
        loser_name = str(row.get("loser_name", "")).strip()
        if not winner_name or not loser_name:
            return

        # Parse date
        date_str = str(row.get("tourney_date", ""))
        try:
            match_date = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            return

        surface_raw = str(row.get("surface", "Hard")).lower()
        surface_map = {"hard": "hard", "clay": "clay", "grass": "grass"}
        surface = surface_map.get(surface_raw, "hard")

        winner = self._get_or_create_player(winner_name)
        loser = self._get_or_create_player(loser_name)

        # Update player metadata if available
        for player, prefix in [(winner, "winner"), (loser, "loser")]:
            if not player.ranking:
                rank = row.get(f"{prefix}_rank")
                if pd.notna(rank):
                    player.ranking = int(rank)
            if not player.age:
                age = row.get(f"{prefix}_age")
                if pd.notna(age):
                    player.age = float(age)
            if not player.height_cm:
                ht = row.get(f"{prefix}_ht")
                if pd.notna(ht):
                    player.height_cm = int(ht)
            if not player.handedness:
                hand = row.get(f"{prefix}_hand")
                if pd.notna(hand):
                    player.handedness = "left" if str(hand).upper() == "L" else "right"

        # Tournament
        tourney_name = str(row.get("tourney_name", "Unknown"))
        tournament = self.db.query(Tournament).filter(
            Tournament.name == tourney_name,
            Tournament.year == match_date.year,
        ).first()
        if not tournament:
            tournament = Tournament(
                name=tourney_name,
                surface=surface,
                year=match_date.year,
                start_date=match_date.date(),
            )
            self.db.add(tournament)
            self.db.commit()
            self.db.refresh(tournament)

        # Create external ID
        ext_id = f"hist_{row.get('tourney_id','')}_{row.get('match_num','')}"
        existing = self.db.query(Match).filter(Match.external_id == ext_id).first()
        if existing:
            return

        import random
        if random.random() > 0.5:
            p1, p2 = winner, loser
            p1_rank = int(row["winner_rank"]) if pd.notna(row.get("winner_rank")) else None
            p2_rank = int(row["loser_rank"])  if pd.notna(row.get("loser_rank"))  else None
        else:
            p1, p2 = loser, winner
            p1_rank = int(row["loser_rank"])  if pd.notna(row.get("loser_rank"))  else None
            p2_rank = int(row["winner_rank"]) if pd.notna(row.get("winner_rank")) else None

        match = Match(
            external_id=ext_id,
            tournament_id=tournament.id,
            player1_id=p1.id,
            player2_id=p2.id,
            winner_id=winner.id,
            round=str(row.get("round", "")),
            surface=surface,
            scheduled_at=match_date,
            completed_at=match_date,
            is_completed=True,
            score=str(row.get("score", "")),
            player1_rank=p1_rank,
            player2_rank=p2_rank,
        )
        self.db.add(match)
