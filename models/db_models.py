from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Date,
    ForeignKey, Text, JSON, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from database import Base


# ── Python enums (stored as String in SQLite) ─────────────────

class Surface(str, enum.Enum):
    HARD    = "hard"
    CLAY    = "clay"
    GRASS   = "grass"
    CARPET  = "carpet"


class TourLevel(str, enum.Enum):
    ATP        = "atp"
    WTA        = "wta"
    CHALLENGER = "challenger"
    ITF        = "itf"


class BetStatus(str, enum.Enum):
    PENDING   = "pending"
    WON       = "won"
    LOST      = "lost"
    VOID      = "void"
    CANCELLED = "cancelled"


class StakingMethod(str, enum.Enum):
    KELLY           = "kelly"
    FRACTIONAL_KELLY= "fractional_kelly"
    FLAT            = "flat"


# ── Players ───────────────────────────────────────────────────

class Player(Base):
    __tablename__ = "players"

    id              = Column(Integer, primary_key=True)
    external_id     = Column(String(100), unique=True, nullable=True)
    name            = Column(String(200), nullable=False)
    first_name      = Column(String(100))
    last_name       = Column(String(100))
    country         = Column(String(3))
    tour            = Column(String(20))
    ranking         = Column(Integer)
    ranking_points  = Column(Integer)
    age             = Column(Float)
    height_cm       = Column(Integer)
    weight_kg       = Column(Integer)
    handedness      = Column(String(10))
    turned_pro      = Column(Integer)
    is_active       = Column(Boolean, default=True)
    nationality     = Column(String(100))
    created_at      = Column(DateTime, server_default=func.now())
    updated_at      = Column(DateTime, onupdate=func.now())

    elo_ratings    = relationship("EloRating",    back_populates="player")
    matches_as_p1  = relationship("Match", foreign_keys="Match.player1_id", back_populates="player1")
    matches_as_p2  = relationship("Match", foreign_keys="Match.player2_id", back_populates="player2")
    stats          = relationship("PlayerStats",  back_populates="player")

    __table_args__ = (
        Index("idx_player_name",    "name"),
        Index("idx_player_ranking", "ranking"),
    )


# ── Tournaments ───────────────────────────────────────────────

class Tournament(Base):
    __tablename__ = "tournaments"

    id          = Column(Integer, primary_key=True)
    external_id = Column(String(100), unique=True, nullable=True)
    name        = Column(String(200), nullable=False)
    tour        = Column(String(20))
    surface     = Column(String(20))
    category    = Column(String(50))
    country     = Column(String(3))
    city        = Column(String(100))
    prize_money = Column(Integer)
    draw_size   = Column(Integer)
    start_date  = Column(Date)
    end_date    = Column(Date)
    year        = Column(Integer)
    created_at  = Column(DateTime, server_default=func.now())

    matches = relationship("Match", back_populates="tournament")


# ── Matches ───────────────────────────────────────────────────

class Match(Base):
    __tablename__ = "matches"

    id            = Column(Integer, primary_key=True)
    external_id   = Column(String(200), unique=True, nullable=True)
    tournament_id = Column(Integer, ForeignKey("tournaments.id"))
    player1_id    = Column(Integer, ForeignKey("players.id"), nullable=False)
    player2_id    = Column(Integer, ForeignKey("players.id"), nullable=False)
    winner_id     = Column(Integer, ForeignKey("players.id"), nullable=True)

    round         = Column(String(50))
    surface       = Column(String(20))
    scheduled_at  = Column(DateTime)
    completed_at  = Column(DateTime, nullable=True)
    is_completed  = Column(Boolean, default=False)
    is_walkover   = Column(Boolean, default=False)
    is_retired    = Column(Boolean, default=False)

    score           = Column(String(100))
    sets_player1    = Column(Integer)
    sets_player2    = Column(Integer)
    player1_rank    = Column(Integer)
    player2_rank    = Column(Integer)
    player1_rank_points = Column(Integer)
    player2_rank_points = Column(Integer)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    tournament           = relationship("Tournament",     back_populates="matches")
    player1              = relationship("Player", foreign_keys=[player1_id], back_populates="matches_as_p1")
    player2              = relationship("Player", foreign_keys=[player2_id], back_populates="matches_as_p2")
    odds                 = relationship("MatchOdds",          back_populates="match")
    predictions          = relationship("ModelPrediction",    back_populates="match")
    betting_opportunities= relationship("BettingOpportunity", back_populates="match")

    __table_args__ = (
        Index("idx_match_scheduled",  "scheduled_at"),
        Index("idx_match_completed",  "is_completed"),
    )


# ── Odds ──────────────────────────────────────────────────────

class MatchOdds(Base):
    __tablename__ = "match_odds"

    id              = Column(Integer, primary_key=True)
    match_id        = Column(Integer, ForeignKey("matches.id"), nullable=False)
    bookmaker       = Column(String(100), nullable=False)
    market          = Column(String(50), default="h2h")
    odds_player1    = Column(Float)
    odds_player2    = Column(Float)
    implied_prob_p1 = Column(Float)
    implied_prob_p2 = Column(Float)
    is_opening      = Column(Boolean, default=False)
    is_closing      = Column(Boolean, default=False)
    timestamp       = Column(DateTime, server_default=func.now())

    match = relationship("Match", back_populates="odds")


# ── Elo Ratings ───────────────────────────────────────────────

class EloRating(Base):
    __tablename__ = "elo_ratings"

    id             = Column(Integer, primary_key=True)
    player_id      = Column(Integer, ForeignKey("players.id"), nullable=False)
    date           = Column(Date, nullable=False)
    surface        = Column(String(20), nullable=True)
    rating         = Column(Float, nullable=False, default=1500.0)
    matches_played = Column(Integer, default=0)

    player = relationship("Player", back_populates="elo_ratings")

    __table_args__ = (
        UniqueConstraint("player_id", "date", "surface", name="uq_elo_player_date_surface"),
        Index("idx_elo_player_date", "player_id", "date"),
    )


# ── Player Stats ──────────────────────────────────────────────

class PlayerStats(Base):
    __tablename__ = "player_stats"

    id                      = Column(Integer, primary_key=True)
    player_id               = Column(Integer, ForeignKey("players.id"), nullable=False)
    surface                 = Column(String(20), nullable=True)
    season                  = Column(Integer)
    matches_played          = Column(Integer, default=0)
    matches_won             = Column(Integer, default=0)
    ace_rate                = Column(Float)
    double_fault_rate       = Column(Float)
    first_serve_pct         = Column(Float)
    first_serve_win_pct     = Column(Float)
    second_serve_win_pct    = Column(Float)
    break_point_saved_pct   = Column(Float)
    return_points_won_pct   = Column(Float)
    tiebreak_win_pct        = Column(Float)
    updated_at              = Column(DateTime, onupdate=func.now())

    player = relationship("Player", back_populates="stats")

    __table_args__ = (
        UniqueConstraint("player_id", "surface", "season", name="uq_stats_player_surface_season"),
    )


# ── Model Predictions ─────────────────────────────────────────

class ModelPrediction(Base):
    __tablename__ = "model_predictions"

    id               = Column(Integer, primary_key=True)
    match_id         = Column(Integer, ForeignKey("matches.id"), nullable=False)
    model_version    = Column(String(50), nullable=False)
    model_name       = Column(String(100))
    prob_player1     = Column(Float, nullable=False)
    prob_player2     = Column(Float, nullable=False)
    confidence       = Column(Float)
    feature_snapshot = Column(JSON)
    predicted_at     = Column(DateTime, server_default=func.now())

    match        = relationship("Match",              back_populates="predictions")
    opportunities= relationship("BettingOpportunity", back_populates="prediction")

    __table_args__ = (
        Index("idx_prediction_match", "match_id"),
    )


# ── Betting Opportunities ─────────────────────────────────────

class BettingOpportunity(Base):
    __tablename__ = "betting_opportunities"

    id                    = Column(Integer, primary_key=True)
    match_id              = Column(Integer, ForeignKey("matches.id"), nullable=False)
    prediction_id         = Column(Integer, ForeignKey("model_predictions.id"), nullable=False)
    bet_on_player         = Column(Integer)
    bookmaker             = Column(String(100))
    decimal_odds          = Column(Float)
    implied_probability   = Column(Float)
    model_probability     = Column(Float)
    edge_percent          = Column(Float)
    expected_value        = Column(Float)
    kelly_stake_pct       = Column(Float)
    recommended_stake_pct = Column(Float)
    recommended_stake     = Column(Float)
    confidence            = Column(Float)
    is_alerted            = Column(Boolean, default=False)
    alert_sent_at         = Column(DateTime, nullable=True)
    detected_at           = Column(DateTime, server_default=func.now())
    created_at            = Column(DateTime, server_default=func.now())

    match       = relationship("Match",           back_populates="betting_opportunities")
    prediction  = relationship("ModelPrediction", back_populates="opportunities")
    placed_bets = relationship("PlacedBet",       back_populates="opportunity")

    __table_args__ = (
        Index("idx_opp_edge",     "edge_percent"),
        Index("idx_opp_detected", "detected_at"),
    )


# ── Placed Bets ───────────────────────────────────────────────

class PlacedBet(Base):
    __tablename__ = "placed_bets"

    id             = Column(Integer, primary_key=True)
    opportunity_id = Column(Integer, ForeignKey("betting_opportunities.id"), nullable=True)
    stake_amount   = Column(Float, nullable=False)
    staking_method = Column(String(30), default="fractional_kelly")
    bookmaker      = Column(String(100))
    odds_taken     = Column(Float)
    status         = Column(String(20), default="pending")
    profit_loss    = Column(Float, nullable=True)
    clv            = Column(Float, nullable=True)
    placed_at      = Column(DateTime, server_default=func.now())
    settled_at     = Column(DateTime, nullable=True)
    notes          = Column(Text)

    opportunity = relationship("BettingOpportunity", back_populates="placed_bets")

    __table_args__ = (
        Index("idx_bet_status",    "status"),
        Index("idx_bet_placed_at", "placed_at"),
    )


# ── Model Registry ────────────────────────────────────────────

class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id               = Column(Integer, primary_key=True)
    version          = Column(String(50), unique=True, nullable=False)
    name             = Column(String(100))
    algorithm        = Column(String(50))
    tour             = Column(String(20))
    surface          = Column(String(20), nullable=True)
    log_loss         = Column(Float)
    brier_score      = Column(Float)
    roc_auc          = Column(Float)
    calibration_error= Column(Float)
    n_train_samples  = Column(Integer)
    n_features       = Column(Integer)
    feature_importance = Column(JSON)
    hyperparameters  = Column(JSON)
    train_date_start = Column(Date)
    train_date_end   = Column(Date)
    is_active        = Column(Boolean, default=False)
    file_path        = Column(String(500))
    created_at       = Column(DateTime, server_default=func.now())
