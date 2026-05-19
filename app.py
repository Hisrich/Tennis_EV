"""
Tennis EV Platform
Run with: python app.py
"""
import os
import threading
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, abort, send_from_directory
from flask_cors import CORS
from sqlalchemy import desc
from loguru import logger

from config import config
from database import db_session, engine
from models.db_models import (
    Base, Match, Player, BettingOpportunity, PlacedBet,
    ModelPrediction, ModelRegistry, EloRating, MatchOdds, BetStatus
)

# ── Create DB tables ──────────────────────────────────────────
Base.metadata.create_all(bind=engine)
os.makedirs(config.MODEL_DIR, exist_ok=True)

# ── Flask app ─────────────────────────────────────────────────
app = Flask(__name__, static_folder="static")
CORS(app)


# ── Serve frontend HTML pages ─────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")

@app.route("/<path:filename>")
def frontend(filename):
    # Serve any file from frontend/ (html, css, js)
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
    if os.path.exists(os.path.join(frontend_path, filename)):
        return send_from_directory("frontend", filename)
    return send_from_directory("frontend", "index.html")


# ── Serializers ───────────────────────────────────────────────

def ser_opportunity(opp):
    match = opp.match
    return {
        "id": opp.id,
        "match_id": opp.match_id,
        "bet_on_player": opp.bet_on_player,
        "bookmaker": opp.bookmaker,
        "decimal_odds": opp.decimal_odds,
        "implied_probability": opp.implied_probability,
        "model_probability": opp.model_probability,
        "edge_percent": opp.edge_percent,
        "expected_value": opp.expected_value,
        "kelly_stake_pct": opp.kelly_stake_pct,
        "recommended_stake": opp.recommended_stake,
        "confidence": opp.confidence,
        "is_alerted": opp.is_alerted,
        "created_at": opp.created_at.isoformat() if opp.created_at else None,
        "match": {
            "id": match.id,
            "surface": match.surface,
            "round": match.round,
            "scheduled_at": match.scheduled_at.isoformat() if match.scheduled_at else None,
            "player1": {"id": match.player1.id, "name": match.player1.name} if match.player1 else None,
            "player2": {"id": match.player2.id, "name": match.player2.name} if match.player2 else None,
            "tournament": {"name": match.tournament.name} if match.tournament else None,
        } if match else None,
    }


def ser_match(m):
    return {
        "id": m.id,
        "surface": m.surface,
        "round": m.round,
        "scheduled_at": m.scheduled_at.isoformat() if m.scheduled_at else None,
        "is_completed": m.is_completed,
        "player1": {"id": m.player1.id, "name": m.player1.name, "ranking": m.player1.ranking} if m.player1 else None,
        "player2": {"id": m.player2.id, "name": m.player2.name, "ranking": m.player2.ranking} if m.player2 else None,
        "tournament": {"id": m.tournament.id, "name": m.tournament.name} if m.tournament else None,
        "winner_id": m.winner_id,
    }


def ser_player(p):
    return {
        "id": p.id,
        "name": p.name,
        "ranking": p.ranking,
        "nationality": p.nationality or p.country,
        "country": p.country,
        "date_of_birth": None,
    }


def ser_bet(b):
    return {
        "id": b.id,
        "opportunity_id": b.opportunity_id,
        "stake_amount": b.stake_amount,
        "bookmaker": b.bookmaker,
        "odds_taken": b.odds_taken,
        "status": b.status,
        "profit_loss": b.profit_loss,
        "placed_at": b.placed_at.isoformat() if b.placed_at else None,
        "settled_at": b.settled_at.isoformat() if b.settled_at else None,
        "notes": b.notes,
    }


def ser_model(m):
    return {
        "id": m.id,
        "version": m.version,
        "algorithm": m.algorithm,
        "is_active": m.is_active,
        "log_loss": m.log_loss,
        "brier_score": m.brier_score,
        "roc_auc": m.roc_auc,
        "n_train_samples": m.n_train_samples,
        "train_date_start": str(m.train_date_start) if m.train_date_start else None,
        "train_date_end": str(m.train_date_end) if m.train_date_end else None,
        "created_at": m.created_at.isoformat() if m.created_at else None,
        "feature_importance": m.feature_importance or {},
    }


# ── Health ────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


# ── Opportunities ─────────────────────────────────────────────

@app.route("/api/opportunities")
def get_opportunities():
    min_edge  = float(request.args.get("min_edge", config.MIN_EDGE_PERCENT))
    limit     = min(int(request.args.get("limit", 50)), 200)
    only_live = request.args.get("only_live", "true").lower() == "true"

    with db_session() as db:
        q = (
            db.query(BettingOpportunity)
            .filter(BettingOpportunity.edge_percent >= min_edge)
            .join(Match)
        )
        if only_live:
            q = q.filter(Match.is_completed == False, Match.scheduled_at > datetime.utcnow())
        opps = q.order_by(desc(BettingOpportunity.edge_percent)).limit(limit).all()
        return jsonify([ser_opportunity(o) for o in opps])


@app.route("/api/opportunities/<int:opp_id>")
def get_opportunity(opp_id):
    with db_session() as db:
        opp = db.query(BettingOpportunity).filter(BettingOpportunity.id == opp_id).first()
        if not opp:
            abort(404, description="Opportunity not found")
        return jsonify(ser_opportunity(opp))


# ── Matches ───────────────────────────────────────────────────

@app.route("/api/matches")
def get_matches():
    upcoming_only = request.args.get("upcoming_only", "false").lower() == "true"
    limit = min(int(request.args.get("limit", 50)), 200)
    with db_session() as db:
        q = db.query(Match)
        if upcoming_only:
            q = q.filter(Match.is_completed == False, Match.scheduled_at > datetime.utcnow())
        return jsonify([ser_match(m) for m in q.order_by(desc(Match.scheduled_at)).limit(limit).all()])


@app.route("/api/matches/<int:match_id>")
def get_match(match_id):
    with db_session() as db:
        m = db.query(Match).filter(Match.id == match_id).first()
        if not m:
            abort(404, description="Match not found")
        return jsonify(ser_match(m))


# ── Performance ───────────────────────────────────────────────

@app.route("/api/performance/summary")
def get_performance_summary():
    days   = int(request.args.get("days", 30))
    cutoff = datetime.utcnow() - timedelta(days=days)
    with db_session() as db:
        bets   = db.query(PlacedBet).filter(PlacedBet.placed_at >= cutoff).all()
        total  = len(bets)
        won    = [b for b in bets if b.status == "won"]
        lost   = [b for b in bets if b.status == "lost"]
        profit = sum(b.profit_loss or 0 for b in bets)
        staked = sum(b.stake_amount for b in bets)
        roi    = profit / staked if staked > 0 else 0
        return jsonify({
            "period_days": days,
            "total_bets": total,
            "wins": len(won),
            "losses": len(lost),
            "pending": total - len(won) - len(lost),
            "win_rate": len(won) / total if total > 0 else 0,
            "total_profit": round(profit, 2),
            "total_staked": round(staked, 2),
            "roi_percent": round(roi * 100, 2),
        })


@app.route("/api/performance/bankroll-history")
def get_bankroll_history():
    days   = int(request.args.get("days", 90))
    cutoff = datetime.utcnow() - timedelta(days=days)
    with db_session() as db:
        bets = (
            db.query(PlacedBet)
            .filter(PlacedBet.placed_at >= cutoff, PlacedBet.profit_loss.isnot(None))
            .order_by(PlacedBet.placed_at.asc())
            .all()
        )
        bankroll = config.DEFAULT_BANKROLL
        history  = [{"date": cutoff.isoformat(), "bankroll": bankroll}]
        for bet in bets:
            bankroll += bet.profit_loss or 0
            history.append({
                "date": (bet.settled_at or bet.placed_at).isoformat(),
                "bankroll": round(bankroll, 2),
            })
        return jsonify(history)


# ── Bets ──────────────────────────────────────────────────────

@app.route("/api/bets", methods=["GET", "POST"])
def bets():
    if request.method == "POST":
        data = request.get_json()
        if not data:
            abort(400, description="JSON body required")
        with db_session() as db:
            opp = db.query(BettingOpportunity).filter(
                BettingOpportunity.id == data.get("opportunity_id")
            ).first()
            if not opp:
                abort(404, description="Opportunity not found")
            bet = PlacedBet(
                opportunity_id=data["opportunity_id"],
                stake_amount=data["stake_amount"],
                bookmaker=data["bookmaker"],
                odds_taken=data["odds_taken"],
                notes=data.get("notes"),
            )
            db.add(bet)
            db.flush()
            return jsonify(ser_bet(bet)), 201

    # GET
    status = request.args.get("status")
    limit  = int(request.args.get("limit", 100))
    with db_session() as db:
        q = db.query(PlacedBet)
        if status:
            q = q.filter(PlacedBet.status == status.lower())
        return jsonify([ser_bet(b) for b in q.order_by(desc(PlacedBet.placed_at)).limit(limit).all()])


@app.route("/api/bets/<int:bet_id>/settle", methods=["PATCH"])
def settle_bet(bet_id):
    data = request.get_json()
    if not data or "status" not in data:
        abort(400, description="status field required")
    with db_session() as db:
        bet = db.query(PlacedBet).filter(PlacedBet.id == bet_id).first()
        if not bet:
            abort(404, description="Bet not found")
        status_val     = data["status"].lower()
        bet.status     = status_val
        bet.settled_at = datetime.utcnow()
        if status_val == "won":
            bet.profit_loss = bet.stake_amount * (bet.odds_taken - 1)
        elif status_val == "lost":
            bet.profit_loss = -bet.stake_amount
        return jsonify({"status": "settled", "profit_loss": bet.profit_loss})


# ── Models ────────────────────────────────────────────────────

@app.route("/api/models")
def list_models():
    with db_session() as db:
        models = db.query(ModelRegistry).order_by(desc(ModelRegistry.created_at)).all()
        return jsonify([ser_model(m) for m in models])


@app.route("/api/models/train", methods=["POST"])
def trigger_training():
    algorithm = request.args.get("algorithm", "xgboost")
    def _train():
        with db_session() as db:
            from services.ml_engine import MLEngine
            try:
                engine = MLEngine(db)
                version = engine.train(algorithm=algorithm)
                logger.info(f"Model retrain complete: {version}")
            except Exception as e:
                logger.error(f"Model retrain failed: {e}")
    threading.Thread(target=_train, daemon=True).start()
    return jsonify({"status": "training_started", "algorithm": algorithm})


# ── Players ───────────────────────────────────────────────────

@app.route("/api/players")
def get_players():
    search = request.args.get("search", "")
    limit  = int(request.args.get("limit", 50))
    with db_session() as db:
        q = db.query(Player)
        if search:
            q = q.filter(Player.name.ilike(f"%{search}%"))
        players = q.order_by(Player.ranking.asc().nullslast()).limit(limit).all()
        return jsonify([ser_player(p) for p in players])


@app.route("/api/players/<int:player_id>/elo-history")
def get_player_elo_history(player_id):
    surface = request.args.get("surface")
    with db_session() as db:
        q = db.query(EloRating).filter(EloRating.player_id == player_id)
        if surface:
            q = q.filter(EloRating.surface == surface)
        else:
            q = q.filter(EloRating.surface.is_(None))
        records = q.order_by(EloRating.date.asc()).all()
        return jsonify([{"date": str(r.date), "rating": r.rating} for r in records])


# ── Admin / task triggers ─────────────────────────────────────

def _run_in_thread(fn):
    """Run a task function in a background thread."""
    threading.Thread(target=fn, daemon=True).start()


@app.route("/api/admin/collect-odds", methods=["POST"])
def trigger_odds_collection():
    def _task():
        with db_session() as db:
            from services.data_collection import DataIngestionService
            try:
                svc = DataIngestionService(db)
                count = svc.ingest_odds()
                logger.info(f"collect_odds: {count} records")
            except Exception as e:
                logger.error(f"collect_odds error: {e}")
    _run_in_thread(_task)
    return jsonify({"status": "started"})


@app.route("/api/admin/run-predictions", methods=["POST"])
def trigger_predictions():
    def _task():
        with db_session() as db:
            from services.ml_engine import MLEngine
            try:
                eng = MLEngine(db)
                upcoming = (
                    db.query(Match)
                    .filter(Match.is_completed == False, Match.scheduled_at > datetime.utcnow())
                    .all()
                )
                for match in upcoming:
                    eng.predict_and_save(match)
                logger.info(f"run_predictions: {len(upcoming)} matches")
            except Exception as e:
                logger.error(f"run_predictions error: {e}")
    _run_in_thread(_task)
    return jsonify({"status": "started"})


@app.route("/api/admin/scan-value", methods=["POST"])
def trigger_value_scan():
    def _task():
        with db_session() as db:
            from services.value_engine import ValueBettingEngine
            try:
                eng = ValueBettingEngine(db)
                opps = eng.scan_all_upcoming()
                logger.info(f"scan_for_value: {len(opps)} opportunities")
            except Exception as e:
                logger.error(f"scan_for_value error: {e}")
    _run_in_thread(_task)
    return jsonify({"status": "started"})


@app.route("/api/admin/rebuild-elo", methods=["POST"])
def trigger_elo_rebuild():
    def _task():
        with db_session() as db:
            from services.elo_engine import EloEngine
            try:
                eng = EloEngine(db)
                eng.rebuild_all()
                logger.info("rebuild_elo complete")
            except Exception as e:
                logger.error(f"rebuild_elo error: {e}")
    _run_in_thread(_task)
    return jsonify({"status": "started"})


@app.route("/api/admin/ingest-csv", methods=["POST"])
def ingest_historical_csv():
    data = request.get_json()
    if not data or "file_path" not in data:
        abort(400, description="file_path required")
    with db_session() as db:
        from services.data_collection import DataIngestionService
        svc   = DataIngestionService(db)
        count = svc.ingest_historical_from_csv(data["file_path"])
        return jsonify({"status": "ok", "records_ingested": count})
    

@app.route("/api/admin/upload-file", methods=["POST"])
def upload_and_ingest():
    if "file" not in request.files:
        abort(400, description="No file uploaded")
    f = request.files["file"]
    if not f.filename:
        abort(400, description="Empty filename")

    import tempfile, os
    suffix = os.path.splitext(f.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        with db_session() as db:
            from services.data_collection import DataIngestionService
            count = DataIngestionService(db).ingest_historical_from_csv(tmp_path)
        return jsonify({"status": "ok", "records_ingested": count})
    finally:
        os.unlink(tmp_path)  # delete the temp file when done


# ── Backtest ──────────────────────────────────────────────────

@app.route("/api/backtest", methods=["POST"])
def run_backtest():
    data = request.get_json()
    if not data:
        abort(400, description="JSON body required")
    with db_session() as db:
        from services.backtester import Backtester
        bt = Backtester(
            db=db,
            model_version=data.get("model_version"),
            min_edge_pct=data.get("min_edge_pct", config.MIN_EDGE_PERCENT),
            kelly_fraction=data.get("kelly_fraction", config.KELLY_FRACTION),
            min_confidence=data.get("min_confidence", config.MIN_CONFIDENCE),
            bankroll=data.get("bankroll", config.DEFAULT_BANKROLL),
        )
        start  = datetime.fromisoformat(data["start_date"]) if data.get("start_date") else None
        end    = datetime.fromisoformat(data["end_date"])   if data.get("end_date")   else None
        result = bt.run(start_date=start, end_date=end)
        return jsonify(result.summary())


# ── Stats overview ────────────────────────────────────────────

@app.route("/api/stats/overview")
def get_overview():
    now = datetime.utcnow()
    with db_session() as db:
        active_model = db.query(ModelRegistry).filter(ModelRegistry.is_active == True).first()
        return jsonify({
            "upcoming_matches": db.query(Match).filter(
                Match.is_completed == False, Match.scheduled_at > now
            ).count(),
            "active_opportunities": db.query(BettingOpportunity).join(Match).filter(
                Match.is_completed == False
            ).count(),
            "total_predictions": db.query(ModelPrediction).count(),
            "total_bets": db.query(PlacedBet).count(),
            "active_model": active_model.version if active_model else None,
        })


# ── Error handlers ────────────────────────────────────────────

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": str(e.description)}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": str(e.description)}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ── Background scheduler ──────────────────────────────────────

def start_scheduler():
    from apscheduler.schedulers.background import BackgroundScheduler

    scheduler = BackgroundScheduler(timezone="UTC")

    def job_collect_odds():
        with db_session() as db:
            from services.data_collection import DataIngestionService
            try:
                DataIngestionService(db).ingest_odds()
            except Exception as e:
                logger.error(f"Scheduled collect_odds error: {e}")

    def job_run_predictions():
        with db_session() as db:
            from services.ml_engine import MLEngine
            try:
                eng = MLEngine(db)
                model = eng._load_active_model()
                if model is None:
                    logger.error("run_predictions: no active model, skipping")
                    return
                upcoming = db.query(Match).filter(
                    Match.is_completed == False,
                    Match.scheduled_at > datetime.utcnow()
                ).all()
                logger.info(f"run_predictions: found {len(upcoming)} upcoming matches")
                saved = 0
                for match in upcoming:
                    result = eng.predict_and_save(match)
                    if result:
                        saved += 1
                logger.info(f"run_predictions: saved {saved}/{len(upcoming)} predictions")
            except Exception as e:
                logger.error(f"Scheduled run_predictions error: {e}", exc_info=True)

    def job_scan_value():
        with db_session() as db:
            from services.value_engine import ValueBettingEngine
            try:
                ValueBettingEngine(db).scan_all_upcoming()
            except Exception as e:
                logger.error(f"Scheduled scan_value error: {e}")

    def job_send_alerts():
        with db_session() as db:
            from services.telegram_bot import AlertService
            try:
                AlertService(db).send_pending_alerts()
            except Exception as e:
                logger.error(f"Scheduled send_alerts error: {e}")

    def job_update_elo():
        with db_session() as db:
            from services.elo_engine import EloEngine
            try:
                cutoff  = datetime.utcnow() - timedelta(hours=7)
                recent  = db.query(Match).filter(
                    Match.is_completed == True,
                    Match.winner_id.isnot(None),
                    Match.completed_at >= cutoff,
                ).all()
                eng = EloEngine(db)
                for match in recent:
                    eng.process_match(match, save=True)
            except Exception as e:
                logger.error(f"Scheduled update_elo error: {e}")

    def job_daily_summary():
        with db_session() as db:
            from services.telegram_bot import AlertService
            try:
                AlertService(db).send_daily_summary(db)
            except Exception as e:
                logger.error(f"Scheduled daily_summary error: {e}")

    def job_retrain():
        with db_session() as db:
            from services.ml_engine import MLEngine
            try:
                MLEngine(db).train(algorithm="xgboost")
                logger.info("Weekly retrain complete")
            except Exception as e:
                logger.error(f"Scheduled retrain error: {e}")

    scheduler.add_job(job_collect_odds,    "interval", hours=6,  id="collect_odds")
    scheduler.add_job(job_run_predictions, "interval", minutes=30,  id="run_predictions")
    scheduler.add_job(job_scan_value,      "interval", minutes=30,  id="scan_value")
    scheduler.add_job(job_send_alerts,     "interval", minutes=15,  id="send_alerts")
    scheduler.add_job(job_update_elo,      "interval", hours=6,     id="update_elo")
    scheduler.add_job(job_daily_summary,   "cron",     hour=23, minute=0, id="daily_summary")
    scheduler.add_job(job_retrain,         "cron",     day_of_week="sun", hour=2, minute=0, id="retrain")

    scheduler.start()
    logger.info("Background scheduler started (collect/predict/scan every 6hours, alerts every 15min)")
    return scheduler


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    logger.info(f"Starting Tennis EV Platform on http://localhost:{config.PORT}")
    logger.info(f"Database: {config.DATABASE_URL}")

    start_scheduler()

    app.run(
        host="0.0.0.0",
        port=config.PORT,
        debug=config.DEBUG,
        use_reloader=False,   # reloader would double-start the scheduler
    )
