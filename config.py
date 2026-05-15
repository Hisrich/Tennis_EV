import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    DATABASE_URL      = os.getenv("DATABASE_URL", "sqlite:///tennis_ev.db")
    SECRET_KEY        = os.getenv("SECRET_KEY", "dev-secret-key")
    DEBUG             = os.getenv("DEBUG", "true").lower() == "true"
    PORT              = int(os.getenv("PORT", 5000))

    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

    ODDS_API_KEY      = os.getenv("ODDS_API_KEY", "")
    ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

    MIN_EDGE_PERCENT  = float(os.getenv("MIN_EDGE_PERCENT", "5.0"))
    MIN_CONFIDENCE    = float(os.getenv("MIN_CONFIDENCE", "0.55"))
    KELLY_FRACTION    = float(os.getenv("KELLY_FRACTION", "0.25"))
    MAX_KELLY_FRACTION= float(os.getenv("MAX_KELLY_FRACTION", "0.25"))
    DEFAULT_BANKROLL  = float(os.getenv("BANKROLL", "1000.0"))

    MIN_MATCHES_FOR_TRAINING = int(os.getenv("MIN_MATCHES_FOR_TRAINING", "200"))
    MODEL_DIR         = os.getenv("MODEL_DIR", "ml_models")


config = Config()
