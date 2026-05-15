"""
Telegram Alert Bot
Sends value bet alerts and daily performance summaries.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List
from loguru import logger

try:
    import telegram
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

from sqlalchemy.orm import Session
from config import config as settings
from models.db_models import BettingOpportunity, Match, Player, PlacedBet, BetStatus


def format_alert(opp: BettingOpportunity, match: Match) -> str:
    """Format a value bet opportunity as a Telegram message."""
    p1 = match.player1.name if match.player1 else "Player 1"
    p2 = match.player2.name if match.player2 else "Player 2"
    bet_player = p1 if opp.bet_on_player == 1 else p2

    tournament = match.tournament.name if match.tournament else "Unknown Tournament"
    surface = match.surface.value.capitalize() if match.surface else "Unknown"
    start_time = match.scheduled_at.strftime("%Y-%m-%d %H:%M UTC") if match.scheduled_at else "TBD"

    ev_pct = opp.expected_value * 100

    lines = [
        "🎾 *VALUE BET FOUND*",
        "",
        f"*Match:* {p1} vs {p2}",
        f"*Bet on:* {bet_player}",
        "",
        f"📊 *Market Analysis*",
        f"  Bookmaker: {opp.bookmaker}",
        f"  Decimal Odds: `{opp.decimal_odds:.2f}`",
        f"  Implied Probability: `{opp.implied_probability*100:.1f}%`",
        f"  Model Probability: `{opp.model_probability*100:.1f}%`",
        "",
        f"💰 *Edge & Value*",
        f"  Edge: `+{opp.edge_percent:.1f}%` ✅",
        f"  Expected Value: `+{ev_pct:.2f}%`",
        f"  Kelly Stake: `{opp.kelly_stake_pct:.1f}%`",
        f"  ➡️ Recommended Stake: `{opp.recommended_stake_pct:.1f}% of bankroll`",
        "",
        f"🏆 *Tournament:* {tournament}",
        f"🎾 *Surface:* {surface}",
        f"⏰ *Start Time:* {start_time}",
        "",
        f"_Alert ID: #{opp.id}_",
    ]
    return "\n".join(lines)


def format_daily_summary(stats: dict) -> str:
    roi_emoji = "📈" if stats.get("roi_percent", 0) >= 0 else "📉"
    lines = [
        "📋 *Daily Performance Summary*",
        f"Date: {datetime.utcnow().strftime('%Y-%m-%d')}",
        "",
        f"🎯 Bets Today: {stats.get('bets_today', 0)}",
        f"✅ Wins: {stats.get('wins', 0)}",
        f"❌ Losses: {stats.get('losses', 0)}",
        f"Win Rate: {stats.get('win_rate', 0)*100:.1f}%",
        "",
        f"{roi_emoji} P&L Today: {stats.get('profit_today', 0):+.2f}",
        f"ROI: {stats.get('roi_percent', 0):+.1f}%",
        "",
        f"💼 Active Opportunities: {stats.get('active_opps', 0)}",
        f"Avg Edge: {stats.get('avg_edge', 0):.1f}%",
    ]
    return "\n".join(lines)


class TelegramAlertBot:
    """
    Sends alerts via Telegram Bot API.
    Falls back to logging if credentials not configured.
    """

    def __init__(self):
        self.token = settings.TELEGRAM_BOT_TOKEN
        self.chat_id = settings.TELEGRAM_CHAT_ID
        self._bot = None

    def _get_bot(self):
        if not TELEGRAM_AVAILABLE:
            return None
        if not self.token:
            return None
        if self._bot is None:
            self._bot = telegram.Bot(token=self.token)
        return self._bot

    def send_message(self, text: str) -> bool:
        """Send a message. Returns True if successful."""
        bot = self._get_bot()
        if not bot or not self.chat_id:
            logger.info(f"[TELEGRAM MOCK] {text[:200]}...")
            return True

        try:
            asyncio.run(bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode="Markdown",
            ))
            return True
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False

    def send_value_alert(self, opp: BettingOpportunity, match: Match) -> bool:
        message = format_alert(opp, match)
        return self.send_message(message)

    def send_daily_summary(self, stats: dict) -> bool:
        message = format_daily_summary(stats)
        return self.send_message(message)


class AlertService:
    """
    Processes new betting opportunities and sends alerts.
    Runs via Celery task.
    """

    def __init__(self, db: Session):
        self.db = db
        self.bot = TelegramAlertBot()

    def send_pending_alerts(self) -> int:
        """Find unalerted opportunities and send Telegram messages."""
        pending = (
            self.db.query(BettingOpportunity)
            .filter(BettingOpportunity.is_alerted == False)
            .join(Match)
            .filter(Match.is_completed == False)
            .all()
        )

        sent = 0
        for opp in pending:
            match = opp.match
            success = self.bot.send_value_alert(opp, match)
            if success:
                opp.is_alerted = True
                opp.alert_sent_at = datetime.utcnow()
                sent += 1

        self.db.commit()
        logger.info(f"Sent {sent} Telegram alerts")
        return sent

    def send_daily_summary(self, db: Session) -> bool:
        """Compile and send daily performance stats."""
        today = datetime.utcnow().date()
        yesterday = today - timedelta(days=1)

        # Settled bets today
        bets = (
            db.query(PlacedBet)
            .filter(PlacedBet.placed_at >= yesterday)
            .all()
        )

        won = [b for b in bets if b.status == BetStatus.WON]
        lost = [b for b in bets if b.status == BetStatus.LOST]
        profit = sum(b.profit_loss or 0 for b in bets)
        staked = sum(b.stake_amount for b in bets)
        roi = profit / staked if staked > 0 else 0

        # Active opportunities
        active_opps = (
            db.query(BettingOpportunity)
            .join(Match)
            .filter(Match.is_completed == False)
            .count()
        )

        active_opps_list = (
            db.query(BettingOpportunity)
            .join(Match)
            .filter(Match.is_completed == False)
            .all()
        )
        avg_edge = sum(o.edge_percent for o in active_opps_list) / len(active_opps_list) if active_opps_list else 0

        stats = {
            "bets_today": len(bets),
            "wins": len(won),
            "losses": len(lost),
            "win_rate": len(won) / len(bets) if bets else 0,
            "profit_today": profit,
            "roi_percent": roi * 100,
            "active_opps": active_opps,
            "avg_edge": avg_edge,
        }

        return self.bot.send_daily_summary(stats)
