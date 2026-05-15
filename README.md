# Tennis EV Platform

A quantitative tennis betting platform that identifies **positive expected value (EV)** opportunities. Finds matches where the bookmaker's implied probability is lower than the model's predicted probability — and sizes stakes using the Kelly Criterion.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11, Flask |
| Database | SQLite (built-in, zero install) |
| ORM | SQLAlchemy |
| ML | XGBoost, LightGBM, scikit-learn |
| Scheduling | APScheduler (runs inside the app) |
| Frontend | HTML, CSS, Vanilla JavaScript |
| Alerts | Telegram (optional) |

No Docker. No PostgreSQL. No Redis. No Node.js.

---

## Project Structure

```
tennis-ev-platform/
├── app.py               ← Run this to start everything
├── config.py            ← Settings loaded from .env
├── database.py          ← SQLAlchemy engine (SQLite by default)
├── seed.py              ← Populate demo data (run once)
├── requirements.txt
├── .env.example
│
├── models/
│   └── db_models.py     ← All database tables
│
├── services/
│   ├── elo_engine.py        ← Elo ratings (overall + per surface)
│   ├── feature_engineering.py  ← 55+ pre-match ML features
│   ├── ml_engine.py         ← Train, calibrate, version models
│   ├── value_engine.py      ← Edge %, EV, Kelly stake sizing
│   ├── backtester.py        ← Historical strategy simulation
│   ├── data_collection.py   ← Odds API + CSV importer
│   └── telegram_bot.py      ← Alert formatting and delivery
│
├── frontend/
│   ├── index.html           ← Dashboard
│   ├── opportunities.html   ← Value bets + record bet modal
│   ├── bets.html            ← Bet tracker + settle buttons
│   ├── performance.html     ← ROI, bankroll curve, bet history
│   ├── backtest.html        ← Backtest config + results
│   ├── model.html           ← Model metrics + feature importance
│   ├── players.html         ← Player search + Elo history
│   ├── admin.html           ← Task triggers + health check
│   ├── css/style.css        ← Dark theme design system
│   └── js/api.js            ← Fetch API client + canvas charts
│
└── ml_models/               ← Trained model files (auto-created)
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment (optional)

```bash
cp .env.example .env
```

The defaults work out of the box — SQLite database, port 5000. Edit `.env` only if you want to change settings or add API keys:

```env
# Everything below is optional
ODDS_API_KEY=your_key        # https://the-odds-api.com (free: 500 req/month)
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
PORT=5000
BANKROLL=1000.0
MIN_EDGE_PERCENT=5.0
```

### 3. Seed demo data

```bash
python seed.py
```

Populates the database with 12 ATP players, Elo ratings, upcoming matches, realistic odds from 3 bookmakers, model predictions, value betting opportunities, and 30 days of historical bets.

### 4. Run the app

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

That's it. The database file (`tennis_ev.db`) is created automatically on first run.

---

## What runs automatically

`app.py` starts a background scheduler alongside Flask. These jobs run while the app is open:

| Job | Interval | What it does |
|-----|----------|--------------|
| Collect odds | Every 30 min | Fetches latest odds from The Odds API |
| Run predictions | Every 30 min | Generates ML win probabilities for upcoming matches |
| Scan for value | Every 30 min | Finds opportunities where edge ≥ threshold |
| Send alerts | Every 15 min | Pushes unalerted opportunities to Telegram |
| Update Elo | Every 6 hours | Recalculates Elo ratings from completed matches |
| Daily summary | 23:00 UTC | Sends P&L summary to Telegram |
| Retrain model | Sunday 02:00 | Full weekly model retrain |

You can also trigger any of these manually from the **System** page in the dashboard.

---

## Pages

| Page | What it shows |
|------|--------------|
| Dashboard | Stats overview, bankroll chart, top value bets |
| Value Bets | All active EV opportunities, filter by edge, record bets |
| My Bets | Bet history, settle won/lost, running P&L |
| Performance | ROI by period, bankroll curve, full bet table |
| Backtester | Run strategy simulations on historical data |
| Model Analytics | Active model metrics, feature importance, version history |
| Players | Search players, view Elo rating history by surface |
| System | Manual task triggers, CSV import, health check |

---

## API Endpoints

All endpoints return JSON. The frontend uses these directly.

```
GET  /health

GET  /api/stats/overview
GET  /api/opportunities              ?min_edge=5&only_live=true
GET  /api/opportunities/<id>
GET  /api/matches                    ?upcoming_only=false
GET  /api/matches/<id>

GET  /api/performance/summary        ?days=30
GET  /api/performance/bankroll-history  ?days=90

GET  /api/bets                       ?status=pending
POST /api/bets                       { opportunity_id, stake_amount, bookmaker, odds_taken }
PATCH /api/bets/<id>/settle          { status: "won" | "lost" }

GET  /api/models
POST /api/models/train               ?algorithm=xgboost

GET  /api/players                    ?search=djokovic
GET  /api/players/<id>/elo-history   ?surface=clay

POST /api/backtest                   { start_date, end_date, min_edge_pct, kelly_fraction, bankroll }

POST /api/admin/collect-odds
POST /api/admin/run-predictions
POST /api/admin/scan-value
POST /api/admin/rebuild-elo
POST /api/admin/ingest-csv           { file_path }
```

---

## Value Betting Logic

### Remove the bookmaker's overround

```
Implied Prob  = 1 / Decimal Odds
Fair Prob     = Implied Prob / sum(all Implied Probs)   # strips vig
```

### Calculate edge

```
Edge (%) = (Model Probability - Fair Implied Probability) × 100
```

### Expected Value

```
EV = (Model Prob × (Odds - 1)) - (1 - Model Prob)
```

### Kelly Criterion stake

```
f* = (b × p - q) / b      where b = Odds - 1, p = model prob, q = 1 - p
Stake = f* × KELLY_FRACTION × bankroll
```

`KELLY_FRACTION` defaults to `0.25` (quarter Kelly).

---

## Configuration

All settings have defaults — none are required. Set in `.env`:

```env
DATABASE_URL=sqlite:///tennis_ev.db   # Switch to PostgreSQL if needed
PORT=5000
MIN_EDGE_PERCENT=5.0     # Minimum edge to show as opportunity
MIN_CONFIDENCE=0.55      # Minimum model confidence
KELLY_FRACTION=0.25      # Fractional Kelly multiplier
BANKROLL=1000.0          # Starting bankroll
```

To use PostgreSQL instead of SQLite:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/tennis_ev
```

---

## Importing historical data

To train models on real data, download [Jeff Sackmann's ATP dataset](https://github.com/JeffSackmann/tennis_atp) and import via the System page or:

```bash
curl -X POST http://localhost:5000/api/admin/ingest-csv \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/atp_matches_2023.csv"}'
```

---

## License

MIT
