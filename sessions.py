"""
Forex session awareness — restricts trading to the most liquid hours for each pair.

Session hours in UTC:
  Asian   (Tokyo/Sydney): 22:00 – 09:00  (wraps midnight)
  London:                 08:00 – 17:00
  New York:               13:00 – 22:00
  London/NY overlap:      13:00 – 17:00  (highest liquidity of the week)

Why this matters:
  EUR/USD at 03:00 UTC (dead Asian hours) has wide spreads, thin liquidity,
  and erratic price action. The same setup at 14:00 UTC (London/NY overlap)
  is far more reliable. Session filtering is free alpha.
"""
from datetime import datetime, timezone


# Session ranges [start_hour, end_hour) in UTC
# end < start means the session wraps through midnight
SESSIONS = {
    'asian':    (22, 9),    # Tokyo + Sydney combined
    'london':   (8,  17),
    'new_york': (13, 22),
}

# Which sessions each pair is most active and liquid in
PAIR_SESSIONS = {
    'EUR_USD': ['london', 'new_york'],   # Euro trades in European + US hours
    'GBP_USD': ['london', 'new_york'],   # Sterling most active in London
    'USD_JPY': ['asian',  'london'],     # JPY driven by Tokyo, then London
    'AUD_USD': ['asian',  'new_york'],   # AUD follows Sydney, NY for USD leg
    'EUR_GBP': ['london'],               # Pure European cross
    'USD_CHF': ['london', 'new_york'],   # CHF active in European hours
    'EUR_JPY': ['asian',  'london'],     # Yen cross — Asian + European
    'GBP_JPY': ['asian',  'london'],     # Most volatile cross
}


def _in_session(start: int, end: int, hour: int) -> bool:
    """Check if hour falls in [start, end), handling midnight wrap."""
    if start < end:
        return start <= hour < end
    else:
        return hour >= start or hour < end   # e.g. Asian: 22-23 or 0-8


def is_active(instrument: str, hour_utc: int = None) -> bool:
    """
    Return True if the current hour is within an active session for this pair.
    Uses live UTC hour if hour_utc not provided.
    """
    if hour_utc is None:
        hour_utc = datetime.now(timezone.utc).hour

    active = PAIR_SESSIONS.get(instrument, ['london', 'new_york'])
    return any(_in_session(*SESSIONS[s], hour_utc) for s in active)


def current_session(hour_utc: int = None) -> str:
    """Return a human-readable string of currently active sessions."""
    if hour_utc is None:
        hour_utc = datetime.now(timezone.utc).hour

    active = [name for name, (s, e) in SESSIONS.items()
              if _in_session(s, e, hour_utc)]
    return '+'.join(active) if active else 'off-hours'


def session_info(instrument: str, hour_utc: int = None) -> str:
    """One-liner for logging: 'EUR_USD: london+new_york [ACTIVE]'"""
    if hour_utc is None:
        hour_utc = datetime.now(timezone.utc).hour
    status = 'ACTIVE' if is_active(instrument, hour_utc) else 'INACTIVE'
    sess   = current_session(hour_utc)
    return f"{instrument}: {sess} [{status}]"