# Forex Trading Bot — OANDA / EUR_USD

Rules-based EMA(9/21) + RSI(14) signal engine, filtered by an XGBoost confidence model.
Daily cap: +5% profit target, −3% loss limit.

## Architecture

```
Market data (M15 candles)
        ↓
   Indicators (EMA, RSI)
        ↓
   Signal engine    → BUY / SELL / HOLD
        ↓
   ML filter (XGBoost)  → approve if P(win) >= threshold
        ↓
   Risk guard (daily P&L cap)
        ↓
   Order engine → OANDA REST API
```

## Step-by-step playbook

### Phase 1: Setup (do this first, 15 minutes)

1. **Open an OANDA demo account** at https://www.oanda.com/eu-en/trading/demo/
2. **Generate API credentials**: log in → My Account → Manage API Access → Generate
3. **Note your account ID** (format: `001-004-XXXXXXXX-001`)
4. **Set up the project:**
   ```bash
   cp .env.example .env       # then edit .env with your credentials
   pip install -r requirements.txt
   ```
5. **Disable the ML filter for now** — open `config.py` and set:
   ```python
   ML_FILTER_ENABLED = False
   ```
   We need to train it first; until then the bot runs in pure rules mode.

### Phase 2: Build the training dataset (one-off, ~5 minutes)

```bash
python backtest.py
```

This fetches roughly 2 years of EUR/USD M15 candles from OANDA, replays the signal logic across all of them, labels each fired signal with its outcome (TP-first = 1, SL-first = 0), and saves the result to `training_data/EUR_USD_signals.csv`.

**What to look for in the output:**
- Total labeled signals — should be 300–800 over 2 years
- Baseline win rate — typically 35–45% for this type of strategy
- If you see fewer than 100 signals, the strategy is too restrictive — loosen the RSI thresholds in `config.py`

### Phase 3: Train the XGBoost model (~2 minutes)

```bash
python train_model.py
```

This trains the classifier with a chronological 75/25 train/validation split, sweeps thresholds from 0.50 to 0.85 to maximize expected P&L, and saves both the model and threshold to `models/`.

**What to look for:**
- **Validation AUC**: above 0.55 means the model has real predictive signal; above 0.65 is excellent. Below 0.52, the features aren't separating wins from losses and the filter won't help.
- **Best threshold P&L vs baseline P&L**: this is the key number. The filtered strategy should beat the unfiltered baseline by a meaningful margin. If it doesn't, the model isn't worth using yet.
- **Top features**: tells you what the model is using. If `hour` and `day_of_week` dominate, the model may have latched onto time effects rather than market structure — be skeptical.

### Phase 4: Enable the filter and run on demo (run for 2–4 weeks)

1. Re-enable the filter — in `config.py` set `ML_FILTER_ENABLED = True`
2. Start the bot:
   ```bash
   python main.py
   ```
3. Monitor `bot.log` for daily P&L and signal-approval rates
4. Check `signal_log.csv` periodically — this is every signal evaluated (approved or skipped). Eventually this becomes new training data.

**What to watch for:**
- The ML filter should reject 30–60% of raw signals. If it's rejecting <10%, the threshold is too loose. If it's rejecting >80%, the threshold is too tight.
- Filtered live win rate should be roughly similar to the validation win rate from training. Big gaps suggest market regime change or overfitting.

### Phase 5: Go live (only when comfortable with demo results)

When demo performance is consistent across multiple weeks:

1. Open a live OANDA account (different from demo)
2. Generate fresh API credentials for the live account
3. In `.env`, change:
   ```
   OANDA_API_KEY=<live key>
   OANDA_ACCOUNT_ID=<live account id>
   OANDA_ENVIRONMENT=live
   ```
4. Start with a small balance (€100–€500). The position sizing is percentage-based — the strategy works the same at any account size.
5. Run for at least a month and compare live results to the demo baseline before considering scaling up.

### Phase 6: Retrain periodically (every 2–3 months)

Markets shift. Re-run the training pipeline regularly:

```bash
python backtest.py     # refresh historical dataset
python train_model.py  # retrain with new data
```

You can also incorporate `signal_log.csv` from live trading into your training set for higher-fidelity data.

## File reference

| File | Purpose |
|------|---------|
| `config.py` | All tunable parameters |
| `broker.py` | OANDA API wrapper |
| `data.py` | Raw candles → DataFrame |
| `indicators.py` | EMA and RSI calculations |
| `strategy.py` | Signal generation, position sizing |
| `risk.py` | Daily P&L cap enforcement |
| `features.py` | 13 features for the ML model |
| `backtest.py` | Historical replay → labeled training set |
| `train_model.py` | XGBoost training + threshold selection |
| `ml_filter.py` | Live confidence filter |
| `main.py` | Scheduler loop, ties it all together |

## Risk disclosure

CFD trading involves substantial risk of loss. The strategy here is illustrative and not financial advice. **Always test extensively on demo before risking real money.** The bot's daily loss cap protects against catastrophic days but does not guarantee profitability.
