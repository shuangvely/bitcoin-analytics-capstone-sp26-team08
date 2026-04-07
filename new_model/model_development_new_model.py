"""Dynamic DCA weight computation — Enhanced Multi-Signal Strategy.

Extends example_1 (MVRV + 200-day MA + PM BTC sentiment) with four new signals
derived from on-chain and Polymarket data:

  Signal 1 — Active Churn Window (on-chain)
      Best accumulation happens when net exchange flow is moderate-positive
      AND active address count is elevated. Not during panic extremes.
      Source columns: FlowInExUSD, FlowOutExUSD, AdrActCnt (CoinMetrics)

  Signal 2 — Macro Event Volatility (Polymarket markets)
      Two sub-signals:
        a) Event proximity dampener: as days-to-resolution shrinks, reduce
           signal volatility (market is pricing in outcome → less uncertainty).
        b) Event activity gate: "No active events" regime elevates downside
           risk → dampen buying. Moderate-high event intensity → allow full signal.
      Source: finance_politics_markets.parquet (end_date, category)

  Signal 3 — Whale / Smart Money Precursor (Polymarket trades)
      Size-weighted directional signal from large trades (>$10k notional).
      Big bets on crypto-bullish outcomes → positive signal, and vice versa.
      Applied with a 7-day EMA to reduce noise.
      Source: finance_politics_trades.parquet + finance_politics_tokens.parquet

  Signal 4 — Polymarket Risk Index Lead Indicator (Polymarket odds history)
      7-day rolling volatility of prediction market probabilities is a
      regime indicator: high risk index coincides with high BTC prices (bull phase).
      Used as a multiplicative regime filter on the MVRV signal — NOT a raw
      buy signal — to prevent leakage. Lagged 1 day strictly.
      Source: finance_politics_odds_history.parquet

All new features are lagged 1 day in precompute_features() to prevent
look-ahead bias. The existing MVRV + MA + PM-BTC-sentiment pipeline from
example_1 is preserved intact and imported directly.

Weight allocation in compute_dynamic_multiplier():
  MVRV value signal     40%  (core valuation anchor, unchanged from example_1)
  200-day MA signal     12%  (trend context, unchanged)
  PM BTC sentiment      8%   (reduced from 20% to make room for new signals)
  Active churn          18%  (Signal 1 — new)
  Macro modifier        multiplicative gate on combined output (Signal 2)
  Whale signal          10%  (Signal 3 — new, additive)
  Risk regime filter    multiplicative gate on MVRV component (Signal 4)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Import base functionality — same pattern as example_1
# ---------------------------------------------------------------------------
from template.prelude_template import load_polymarket_data
from template.model_development_template import (
    _compute_stable_signal,
    allocate_sequential_stable,
    _clean_array,
)

# ---------------------------------------------------------------------------
# Re-use helpers from example_1 that we don't need to duplicate
# ---------------------------------------------------------------------------
from example_1.model_development_example_1 import (
    #load_polymarket_btc_sentiment,  # existing PM BTC market-activity sentiment
    zscore,
    classify_mvrv_zone,
    compute_mvrv_volatility,
    compute_signal_confidence,
    compute_asymmetric_extreme_boost,
    compute_acceleration_modifier,
    compute_adaptive_trend_modifier,
    compute_mean_reversion_pressure,
)

# =============================================================================
# Constants
# =============================================================================

PRICE_COL = "PriceUSD_coinmetrics"
MVRV_COL = "CapMVRVCur"

# On-chain columns (CoinMetrics)
FLOW_IN_COL = "FlowInExUSD"
FLOW_OUT_COL = "FlowOutExUSD"
ADR_ACT_COL = "AdrActCnt"

# Strategy parameters (inherited)
MIN_W = 1e-6
MA_WINDOW = 200
MVRV_GRADIENT_WINDOW = 30
MVRV_ROLLING_WINDOW = 365
MVRV_ACCEL_WINDOW = 14
DYNAMIC_STRENGTH = 5.0

# MVRV zone thresholds (inherited from example_1)
MVRV_ZONE_DEEP_VALUE = -2.0
MVRV_ZONE_VALUE = -1.0
MVRV_ZONE_CAUTION = 1.5
MVRV_ZONE_DANGER = 2.5

# Volatility dampening (inherited)
MVRV_VOLATILITY_WINDOW = 90
MVRV_VOLATILITY_DAMPENING = 0.2

# --- New signal parameters ---

# Signal 1: Active churn
CHURN_NETFLOW_WINDOW = 90       # Rolling window for net-flow z-score
CHURN_ADR_WINDOW = 180          # Rolling window for address-count percentile
CHURN_NETFLOW_CLIP_LO = -1.5    # Asymmetric clip: dampen panic inflows
CHURN_NETFLOW_CLIP_HI = 2.0     # Allow stronger outflow signal
CHURN_NETFLOW_WEIGHT = 0.6      # Within churn composite
CHURN_ADR_WEIGHT = 0.4          # Within churn composite

# Signal 2: Macro event
MACRO_CRYPTO_CATEGORIES = {"crypto", "business", "politics"}
MACRO_EVENT_WINDOW = 30         # Rolling window for event-count percentile
MACRO_NO_EVENT_DAMPENER = 0.75  # Multiply combined signal when no events active
MACRO_PROXIMITY_MIN = 0.60      # Floor for proximity dampener (day before resolution)

# Signal 3: Whale smart money
WHALE_MIN_NOTIONAL = 10000      # USD notional threshold for "big bet"
WHALE_EMA_SPAN = 7              # Smoothing span in days
WHALE_ZSCORE_WINDOW = 90        # Normalisation window
WHALE_CLIP = 2.0                # Clip whale z-score to ±2

# Signal 4: Risk index
RISK_ZSCORE_WINDOW = 90         # Rolling window for risk-index z-score
RISK_VOL_WINDOW = 7             # Days for intra-day prob volatility
RISK_HIGH_THRESHOLD = 1.5       # z-score above = likely bull/elevated regime
RISK_LOW_THRESHOLD = -0.5       # z-score below = likely bear/cheap regime
RISK_HIGH_DAMPEN = 0.82         # Multiply MVRV when in high-risk (bull) regime
RISK_LOW_AMPLIFY = 1.15         # Multiply MVRV when in low-risk (bear) regime

# Combined signal weights (must sum to 1.0 across additive components)
W_MVRV = 0.45
W_MA = 0.12
#W_PM_SENTIMENT = 0.08
W_CHURN = 0.21
W_WHALE = 0.10
# Macro modifier and risk-regime filter are multiplicative, not additive.
# The remaining 0.12 is absorbed by those two gates naturally.

# Feature columns advertised to the framework
FEATS = [
    "price_vs_ma",
    "mvrv_zscore",
    "mvrv_gradient",
    "mvrv_acceleration",
    "mvrv_zone",
    "mvrv_volatility",
    "signal_confidence",
    # remove this feature "polymarket_sentiment",
    # New features
    "churn_signal",
    "macro_modifier",
    "whale_signal",
    "risk_regime_modifier",
]


# =============================================================================
# Signal 1 — Active Churn  (on-chain, CoinMetrics columns)
# =============================================================================


def compute_churn_signal(df: pd.DataFrame) -> pd.Series:
    """Compute the active-churn accumulation signal from on-chain data.

    Logic (from EDA):
      - MVRV Q1 + NetFlow Q2-Q3 produces highest 30-day forward returns.
      - AdrActCnt is nearly uncorrelated with price → carries independent info.
      - Best accumulation: moderate net outflow + elevated active addresses.
      - Extreme net outflow (Q5) had LOWER returns than moderate outflow → clip.

    Args:
        df: DataFrame with FlowInExUSD, FlowOutExUSD, AdrActCnt columns,
            covering the full available history (pre-lag).

    Returns:
        Daily churn_signal series in approximately [-2, 2], higher = more
        accumulation attractive. Not yet lagged — caller must lag.
    """
    missing = [c for c in (FLOW_IN_COL, FLOW_OUT_COL, ADR_ACT_COL) if c not in df.columns]
    if missing:
        logging.warning(f"churn_signal: missing columns {missing}, returning zeros.")
        return pd.Series(0.0, index=df.index)

    # Net flow: positive = net outflow (coins leaving exchanges = bullish)
    net_flow = df[FLOW_OUT_COL] - df[FLOW_IN_COL]

    # Rolling z-score of net flow, asymmetrically clipped
    nf_z = zscore(net_flow, CHURN_NETFLOW_WINDOW).clip(
        CHURN_NETFLOW_CLIP_LO, CHURN_NETFLOW_CLIP_HI
    )

    # Rolling percentile of active addresses → [0, 1]
    # Use rank to avoid look-ahead; min_periods guards early window
    adr_pct = (
        df[ADR_ACT_COL]
        .rolling(CHURN_ADR_WINDOW, min_periods=CHURN_ADR_WINDOW // 4)
        .rank(pct=True)
        .fillna(0.5)
    )

    # Centre adr_pct around 0 so it contributes bidirectionally
    adr_centred = (adr_pct - 0.5) * 2.0  # [-1, 1]

    # Composite: weight net-flow more heavily; add address activity
    churn = nf_z * CHURN_NETFLOW_WEIGHT + adr_centred * CHURN_ADR_WEIGHT

    return churn.fillna(0.0)


# =============================================================================
# Signal 2 — Macro Event Modifier  (Polymarket markets)
# =============================================================================


def load_macro_event_features(markets_df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Derive two macro-event features from Polymarket markets data.

    Sub-signal A — proximity_dampener  [MACRO_PROXIMITY_MIN, 1.0]
        As days_to_resolution shrinks → value decreases toward MACRO_PROXIMITY_MIN.
        Rationale: price variance narrows near resolution (EDA: democratic-nominee
        example). Reduce position-sizing volatility when a major event is imminent.

    Sub-signal B — event_activity_gate  {MACRO_NO_EVENT_DAMPENER, 1.0}
        No active events → elevated downside risk (EDA: No_Event = highest dd prob).
        Q3-Q4 event intensity → lower drawdown probability → full signal allowed.

    Both sub-signals are combined multiplicatively as macro_modifier.
    NOTE: These are already "safe" features (computed from market metadata,
    not outcome prices) and will be lagged 1 day by precompute_features().

    Args:
        markets_df: Pandas DataFrame loaded from finance_politics_markets.parquet.
                    Must have columns: created_at, end_date, category, volume.
        index: Target date index (full CoinMetrics date range).

    Returns:
        DataFrame with columns [proximity_dampener, event_activity_gate,
        macro_modifier] on `index`.
    """
    result = pd.DataFrame(
        {
            "proximity_dampener": 1.0,
            "event_activity_gate": 1.0,
            "macro_modifier": 1.0,
        },
        index=index,
    )

    if markets_df is None or markets_df.empty:
        logging.warning("macro_event_features: markets_df empty, macro_modifier = 1.0 always.")
        return result

    # Keep only crypto-relevant categories
    relevant = markets_df[
        markets_df["category"].str.lower().isin(MACRO_CRYPTO_CATEGORIES)
    ].copy()

    if relevant.empty:
        logging.warning("macro_event_features: no crypto/business/politics markets found.")
        return result

    # Ensure datetime columns
    for col in ("created_at", "end_date"):
        if col in relevant.columns:
            relevant[col] = pd.to_datetime(relevant[col], utc=True, errors="coerce").dt.tz_localize(None)

    relevant = relevant.dropna(subset=["end_date"])

    # ── Sub-signal A: proximity dampener ────────────────────────────────────
    # For each calendar day, find the minimum days-to-resolution across all
    # currently active markets. A short fuse = high proximity = low dampener.
    days_index = index.normalize()

    # Build a Series: for each date, min days remaining among open markets
    def min_days_remaining(date: pd.Timestamp) -> float:
        active = relevant[
            (relevant.get("created_at", pd.Timestamp.min) <= date)
            & (relevant["end_date"] >= date)
        ]
        if active.empty:
            return np.nan
        days = (active["end_date"] - date).dt.days
        return float(days[days >= 0].min()) if (days >= 0).any() else np.nan

    # This loop is O(dates × markets) — acceptable for ~2000 days × a few thousand markets.
    # For very large datasets, vectorise with a merge-asof approach instead.
    min_days = pd.Series(
        [min_days_remaining(d) for d in days_index],
        index=index,
        dtype=float,
    )

    # Sigmoid-style dampener: far away → 1.0, day-before → MACRO_PROXIMITY_MIN
    # Formula: dampener = MACRO_PROXIMITY_MIN + (1 - MIN) * (1 - exp(-days/30))
    dampener_range = 1.0 - MACRO_PROXIMITY_MIN
    proximity_dampener = MACRO_PROXIMITY_MIN + dampener_range * (
        1.0 - np.exp(-min_days.fillna(60) / 30.0)
    )
    proximity_dampener = proximity_dampener.clip(MACRO_PROXIMITY_MIN, 1.0)

    # ── Sub-signal B: event activity gate ───────────────────────────────────
    # Count active markets per day (created_at <= date <= end_date)
    # Use a vectorised merge approach for efficiency
    relevant["date_start"] = relevant["created_at"].dt.normalize()
    relevant["date_end"] = relevant["end_date"].dt.normalize()

    active_count = pd.Series(0, index=index, dtype=int)
    for _, row in relevant.iterrows():
        mask = (index >= row["date_start"]) & (index <= row["date_end"])
        active_count[mask] += 1

    # Rolling percentile of active-market-count (30-day window)
    active_pct = (
        active_count.rolling(MACRO_EVENT_WINDOW, min_periods=1)
        .rank(pct=True)
        .fillna(0.5)
    )

    # Gate: below 10th percentile (near zero events) → dampen
    event_gate = np.where(active_pct < 0.10, MACRO_NO_EVENT_DAMPENER, 1.0)
    event_gate = pd.Series(event_gate, index=index, dtype=float)

    # ── Combine ──────────────────────────────────────────────────────────────
    result["proximity_dampener"] = proximity_dampener
    result["event_activity_gate"] = event_gate
    result["macro_modifier"] = (proximity_dampener * event_gate).clip(
        MACRO_PROXIMITY_MIN * MACRO_NO_EVENT_DAMPENER, 1.0
    )

    return result


# =============================================================================
# Signal 3 — Whale Smart Money  (Polymarket trades)
# =============================================================================


def load_whale_signal(
    trades_df: pd.DataFrame,
    tokens_df: pd.DataFrame,
    markets_df: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """Derive a size-weighted directional signal from large Polymarket trades.

    A "whale" trade is any trade where notional (price × size) exceeds
    WHALE_MIN_NOTIONAL USD. For crypto-relevant markets:
      - BUY on a bullish-labelled outcome → positive signal
      - SELL on a bullish-labelled outcome → negative signal
      - BUY on a bearish-labelled outcome → negative signal (inverse)

    The raw daily directional mass is smoothed with a 7-day EMA then
    z-scored for normalisation.

    Args:
        trades_df: DataFrame from finance_politics_trades.parquet.
                   Columns: timestamp, market_id, token_id, price, size, side.
        tokens_df: DataFrame from finance_politics_tokens.parquet.
                   Columns: market_id, token_id, outcome.
        markets_df: DataFrame from finance_politics_markets.parquet.
                    Columns: market_id, category.
        index: Target date index.

    Returns:
        Daily whale_signal series in approximately [-2, 2], higher = more
        bullish whale conviction. Not yet lagged — caller must lag.
    """
    neutral = pd.Series(0.0, index=index, name="whale_signal")

    if trades_df is None or trades_df.empty:
        logging.warning("whale_signal: trades_df empty, returning zeros.")
        return neutral

    trades = trades_df.copy()
    tokens = tokens_df.copy() if tokens_df is not None else pd.DataFrame()
    markets = markets_df.copy() if markets_df is not None else pd.DataFrame()

    # ── Normalise timestamps ─────────────────────────────────────────────────
    trades["date"] = pd.to_datetime(trades["timestamp"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    trades = trades.dropna(subset=["date", "price", "size", "side"])

    # ── Compute notional and keep only whale trades ──────────────────────────
    trades["notional"] = trades["price"] * trades["size"]
    whale_trades = trades[trades["notional"] >= WHALE_MIN_NOTIONAL].copy()

    if whale_trades.empty:
        logging.warning("whale_signal: no trades above WHALE_MIN_NOTIONAL threshold.")
        return neutral

    # ── Filter to crypto-relevant markets ────────────────────────────────────
    if not markets.empty and "category" in markets.columns:
        crypto_market_ids = set(
            markets[markets["category"].str.lower().isin(MACRO_CRYPTO_CATEGORIES)]["market_id"]
        )
        whale_trades = whale_trades[whale_trades["market_id"].isin(crypto_market_ids)]

    if whale_trades.empty:
        logging.warning("whale_signal: no whale trades in crypto-relevant markets.")
        return neutral

    # ── Assign outcome polarity ───────────────────────────────────────────────
    # Bullish outcome keywords: "yes", "above", "higher", "over", "bull", "moon"
    # Bearish outcome keywords: "no", "below", "lower", "under", "bear"
    BULLISH_KEYWORDS = {"yes", "above", "higher", "over", "bull", "up", "moon", "win"}
    BEARISH_KEYWORDS = {"no", "below", "lower", "under", "bear", "down", "lose", "fall"}

    if not tokens.empty and "outcome" in tokens.columns:
        tokens["outcome_lower"] = tokens["outcome"].str.lower()
        tokens["is_bullish"] = tokens["outcome_lower"].apply(
            lambda o: 1 if any(k in o for k in BULLISH_KEYWORDS)
            else (-1 if any(k in o for k in BEARISH_KEYWORDS) else 0)
        )
        whale_trades = whale_trades.merge(
            tokens[["token_id", "is_bullish"]], on="token_id", how="left"
        )
        whale_trades["is_bullish"] = whale_trades["is_bullish"].fillna(0)
    else:
        whale_trades["is_bullish"] = 0  # Can't determine polarity without tokens

    # ── Compute directional mass ──────────────────────────────────────────────
    # side=BUY → +1, side=SELL → -1
    whale_trades["side_sign"] = whale_trades["side"].map({"BUY": 1, "SELL": -1}).fillna(0)

    # Effective direction: side_sign × outcome_polarity
    # If polarity is 0 (unknown), the trade contributes 0 to signal
    whale_trades["direction"] = whale_trades["side_sign"] * whale_trades["is_bullish"]

    # Size-weighted daily net direction
    whale_trades["weighted_dir"] = whale_trades["direction"] * whale_trades["notional"]

    daily_whale = (
        whale_trades.groupby("date")["weighted_dir"].sum()
        .reindex(index, fill_value=0.0)
    )

    # ── Smooth and normalise ──────────────────────────────────────────────────
    whale_ema = daily_whale.ewm(span=WHALE_EMA_SPAN, adjust=False).mean()
    whale_z = zscore(whale_ema.to_frame(name="w"), WHALE_ZSCORE_WINDOW)["w"]
    whale_signal = whale_z.clip(-WHALE_CLIP, WHALE_CLIP).fillna(0.0)

    return whale_signal.rename("whale_signal")


# =============================================================================
# Signal 4 — Risk Index Lead Indicator  (Polymarket odds history)
# =============================================================================


def load_risk_regime_modifier(
    odds_df: pd.DataFrame,
    index: pd.DatetimeIndex,
    tokens_df: pd.DataFrame,  
) -> pd.Series:
    """Compute a regime-aware multiplier from the Polymarket odds-history risk index.

    EDA finding: low to moderate risk co moves with 7D forward BTC price, using volatility of log-odds as a better 
    measure of informational velocity than raw probability difference

    Interpretation for DCA - this is a different or almost opposite finding from initial EDA due to
    change of vol calculation methodology
      Low risk index (Calm) -> Accumulation phase -> Amplify buying (RISK_LOW_AMPLIFY)
      High risk index (Chaos) -> Exhaustion phase -> Dampen buying (RISK_HIGH_DAMPEN)

    This is applied as a MULTIPLICATIVE FILTER on the MVRV value signal component,
    not as an independent additive signal, to avoid spurious alpha claims.

    No look-ahead: uses only daily aggregates that would be available at market
    close on day T, lagged by 1 day in precompute_features().

    Args:
        odds_df: DataFrame from finance_politics_odds_history.parquet.
                 Columns: timestamp, market_id, token_id, price.
        index: Target date index.

    Returns:
        Daily risk_regime_modifier series. Values:
          RISK_LOW_AMPLIFY (~1.15) when in low-risk (bear) regime
          1.0               when risk is neutral
          RISK_HIGH_DAMPEN (~0.82) when in high-risk (bull) regime
        Not yet lagged — caller must lag.
    """
    neutral = pd.Series(1.0, index=index, name="risk_regime_modifier")

    if odds_df is None or odds_df.empty or tokens_df is None or tokens_df.empty:
        logging.warning("risk_regime_modifier: odds_df empty, returning 1.0 always.")
        return neutral
    
    # Join Odds with Tokens to get 'Outcome' labels 
    tokens_sub = tokens_df[["market_id", "token_id", "outcome"]].copy()
    tokens_sub["outcome_lower"] = tokens_sub["outcome"].str.lower()

    # Perform the join
    odds = odds_df.merge(tokens_sub, on=["market_id", "token_id"], how="inner")

    # Filter for 'Yes' or 'Up' to ensure we only look at one side of the binary pair
    odds = odds[odds["outcome_lower"].isin(["yes", "up"])]

    # Normalise timestamps to daily 
    odds["date"] = pd.to_datetime(odds["timestamp"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    odds = odds.dropna(subset=["date", "price"])

    # Logit Transformation & Volatility 
    # Clip to avoid log(0)
    daily_close["p_clipped"] = daily_close["price"].clip(1e-4, 1 - 1e-4)
    daily_close["logit_p"] = np.log(daily_close["p_clipped"] / (1 - daily_close["p_clipped"]))

    # Calculate 7-day rolling logit volatility per market
    daily_close = daily_close.sort_values(["market_id", "date"])
    daily_close["logit_diff"] = daily_close.groupby("market_id")["logit_p"].diff()
    
    daily_close["logit_vol_7d"] = (
        daily_close.groupby("market_id")["logit_diff"]
        .transform(lambda s: s.rolling(RISK_VOL_WINDOW, min_periods=4).std())
    )


    # Cross-market daily mean volatility → the "risk index"
    risk_index = (
        daily_close.groupby("date")["logit_vol_7d"]
        .mean()
        .reindex(index, fill_value=np.nan)
        .ffill()
        .fillna(0.0)
    )

    # ── Rolling z-score normalisation ────────────────────────────────────────
    risk_z = zscore(
        risk_index.to_frame(name="r"), RISK_ZSCORE_WINDOW
    )["r"].clip(-3, 3).fillna(0.0)

    # ── Map to regime modifier ────────────────────────────────────────────────
    modifier = np.where(
        risk_z > RISK_HIGH_THRESHOLD,
        RISK_HIGH_DAMPEN,    # Bull/elevated regime → dampen MVRV buy signal
        np.where(
            risk_z < RISK_LOW_THRESHOLD,
            RISK_LOW_AMPLIFY,  # Bear/early regime → amplify MVRV buy signal
            1.0,               # Neutral zone → no adjustment
        ),
    )

    return pd.Series(modifier, index=index, name="risk_regime_modifier").fillna(1.0)


# =============================================================================
# Feature Engineering — precompute_features()
# =============================================================================


def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features for weight calculation.

    Extends example_1's precompute_features() with four new signals.
    ALL signal columns are lagged 1 day to prevent look-ahead bias.

    Data columns expected in df (CoinMetrics merged dataset):
      Required:  PriceUSD_coinmetrics
      Optional:  CapMVRVCur, FlowInExUSD, FlowOutExUSD, AdrActCnt

    Polymarket data is loaded internally via load_polymarket_data()
    (same pattern as example_1).

    Args:
        df: DataFrame with CoinMetrics columns, DatetimeIndex.

    Returns:
        DataFrame with all feature columns, indexed by date.
        All signal columns are lagged 1 day.
    """
    if PRICE_COL not in df.columns:
        raise KeyError(f"'{PRICE_COL}' not found. Available: {list(df.columns)}")

    # ── Price and MA (unchanged from example_1) ──────────────────────────────
    price = df[PRICE_COL].loc["2010-07-18":].copy()
    ma = price.rolling(MA_WINDOW, min_periods=MA_WINDOW // 2).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        price_vs_ma = ((price / ma) - 1).clip(-1, 1).fillna(0)

    # ── MVRV features (unchanged from example_1) ─────────────────────────────
    if MVRV_COL in df.columns:
        mvrv = df[MVRV_COL].loc[price.index]
        mvrv_z = zscore(mvrv, MVRV_ROLLING_WINDOW).clip(-4, 4)
        gradient_raw = mvrv_z.diff(MVRV_GRADIENT_WINDOW)
        gradient_smooth = gradient_raw.ewm(span=MVRV_GRADIENT_WINDOW, adjust=False).mean()
        mvrv_gradient = np.tanh(gradient_smooth * 2).fillna(0)
        accel_raw = mvrv_gradient.diff(MVRV_ACCEL_WINDOW)
        mvrv_acceleration = np.tanh(
            accel_raw.ewm(span=MVRV_ACCEL_WINDOW, adjust=False).mean() * 3
        ).fillna(0)
        mvrv_zone = pd.Series(classify_mvrv_zone(mvrv_z.values), index=mvrv_z.index)
        mvrv_volatility = compute_mvrv_volatility(mvrv_z, MVRV_VOLATILITY_WINDOW)
    else:
        mvrv_z = pd.Series(0.0, index=price.index)
        mvrv_gradient = pd.Series(0.0, index=price.index)
        mvrv_acceleration = pd.Series(0.0, index=price.index)
        mvrv_zone = pd.Series(0, index=price.index)
        mvrv_volatility = pd.Series(0.5, index=price.index)

    # # ── Existing PM BTC sentiment (unchanged from example_1) ─────────────────
    # try:
    #     pm_btc_df = load_polymarket_btc_sentiment()
    #     polymarket_sentiment = (
    #         pm_btc_df["polymarket_sentiment"].reindex(price.index, fill_value=0.5)
    #         if not pm_btc_df.empty else pd.Series(0.5, index=price.index)
    #     )
    # except Exception as e:
    #     logging.warning(f"PM BTC sentiment unavailable: {e}")
    #     polymarket_sentiment = pd.Series(0.5, index=price.index)

    # ── Signal 1: Active churn ────────────────────────────────────────────────
    # Computed from on-chain columns already in df — no external load needed
    churn_raw = compute_churn_signal(df.loc[price.index])

    # ── Load Polymarket data (shared for signals 2, 3, 4) ────────────────────
    polymarket_data = {}
    try:
        polymarket_data = load_polymarket_data() or {}
    except Exception as e:
        logging.warning(f"Could not load Polymarket data for new signals: {e}")

    markets_df_pd = _to_pandas(polymarket_data.get("markets"))
    trades_df_pd = _to_pandas(polymarket_data.get("trades"))
    tokens_df_pd = _to_pandas(polymarket_data.get("tokens"))
    odds_df_pd = _to_pandas(polymarket_data.get("odds"))

    # ── Signal 2: Macro event modifier ───────────────────────────────────────
    try:
        macro_features = load_macro_event_features(markets_df_pd, price.index)
        macro_modifier_raw = macro_features["macro_modifier"]
    except Exception as e:
        logging.warning(f"macro_modifier failed: {e}")
        macro_modifier_raw = pd.Series(1.0, index=price.index)

    # ── Signal 3: Whale smart money ───────────────────────────────────────────
    try:
        whale_raw = load_whale_signal(trades_df_pd, tokens_df_pd, markets_df_pd, price.index)
    except Exception as e:
        logging.warning(f"whale_signal failed: {e}")
        whale_raw = pd.Series(0.0, index=price.index)

    # ── Signal 4: Risk regime modifier ───────────────────────────────────────
    try:
        risk_raw = load_risk_regime_modifier(odds_df_pd, price.index,tokens_df_pd)
    except Exception as e:
        logging.warning(f"risk_regime_modifier failed: {e}")
        risk_raw = pd.Series(1.0, index=price.index)

    # ── Assemble pre-lag DataFrame ────────────────────────────────────────────
    features = pd.DataFrame(
        {
            PRICE_COL: price,
            "price_ma": ma,
            # Signals 
            "price_vs_ma": price_vs_ma,
            "mvrv_zscore": mvrv_z,
            "mvrv_gradient": mvrv_gradient,
            "mvrv_acceleration": mvrv_acceleration,
            "mvrv_zone": mvrv_zone,
            "mvrv_volatility": mvrv_volatility,
            "signal_confidence": 0.5,       # Computed post-lag below
            #"polymarket_sentiment": polymarket_sentiment,
            # New signals 
            "churn_signal": churn_raw.reindex(price.index, fill_value=0.0),
            "macro_modifier": macro_modifier_raw.reindex(price.index, fill_value=1.0),
            "whale_signal": whale_raw.reindex(price.index, fill_value=0.0),
            "risk_regime_modifier": risk_raw.reindex(price.index, fill_value=1.0),
        },
        index=price.index,
    )

    # ── Lag all signal columns by 1 day (no look-ahead bias) ─────────────────
    signal_cols = [
        "price_vs_ma",
        "mvrv_zscore",
        "mvrv_gradient",
        "mvrv_acceleration",
        "mvrv_zone",
        "mvrv_volatility",
        #"polymarket_sentiment",
        "churn_signal",
        "macro_modifier",
        "whale_signal",
        "risk_regime_modifier",
    ]
    features[signal_cols] = features[signal_cols].shift(1)

    # ── Fill NaN with safe defaults ───────────────────────────────────────────
    features["mvrv_zone"] = features["mvrv_zone"].fillna(0)
    features["mvrv_volatility"] = features["mvrv_volatility"].fillna(0.5)
    #features["polymarket_sentiment"] = features["polymarket_sentiment"].fillna(0.5)
    features["churn_signal"] = features["churn_signal"].fillna(0.0)
    features["macro_modifier"] = features["macro_modifier"].fillna(1.0)
    features["whale_signal"] = features["whale_signal"].fillna(0.0)
    features["risk_regime_modifier"] = features["risk_regime_modifier"].fillna(1.0)
    features = features.fillna(0)

    # ── Signal confidence (uses already-lagged values) ────────────────────────
    features["signal_confidence"] = compute_signal_confidence(
        features["mvrv_zscore"].values,
        features["mvrv_gradient"].values,
        features["price_vs_ma"].values,
    )

    return features


# =============================================================================
# Dynamic Multiplier — compute_dynamic_multiplier()
# =============================================================================


def compute_dynamic_multiplier(
    price_vs_ma: np.ndarray,
    mvrv_zscore: np.ndarray,
    mvrv_gradient: np.ndarray,
    mvrv_acceleration: np.ndarray | None = None,
    mvrv_volatility: np.ndarray | None = None,
    signal_confidence: np.ndarray | None = None,
    #polymarket_sentiment: np.ndarray | None = None,
    # New parameters
    churn_signal: np.ndarray | None = None,
    macro_modifier: np.ndarray | None = None,
    whale_signal: np.ndarray | None = None,
    risk_regime_modifier: np.ndarray | None = None,
) -> np.ndarray:
    """Compute weight multiplier from all signals.

    Signal architecture:
      Additive core (weighted sum → combined):
        MVRV value signal     40%  — core valuation, asymmetric extreme boost
        200-day MA signal     12%  — trend context, adaptive trend modifier
        PM BTC sentiment       8%  — new-market activity on Polymarket
        Active churn          18%  — moderate outflow + active addresses (Signal 1)
        Whale signal          10%  — size-weighted big-bet direction (Signal 3)

      Multiplicative gates (applied to combined after weighting):
        Acceleration modifier — momentum / reversal detection (from example_1)
        Confidence boost      — amplify when signals agree (from example_1)
        Volatility dampening  — dampen in extreme MVRV vol (from example_1)
        Macro modifier        — event-proximity + activity gate (Signal 2)  ← NEW
        Risk regime filter    — applied to MVRV component only (Signal 4)   ← NEW

    Args:
        price_vs_ma: Distance from 200-day MA in [-1, 1]
        mvrv_zscore: MVRV Z-score in [-4, 4]
        mvrv_gradient: MVRV trend direction in [-1, 1]
        mvrv_acceleration: Optional MVRV momentum [-1, 1]
        mvrv_volatility: Optional MVRV vol percentile [0, 1]
        signal_confidence: Optional cross-signal agreement [0, 1]
        polymarket_sentiment: Optional PM BTC market activity [0, 1]
        churn_signal: Optional on-chain active churn score [-2, 2]  (Signal 1)
        macro_modifier: Optional event proximity × activity gate [0.45, 1.0]  (Signal 2)
        whale_signal: Optional whale directional z-score [-2, 2]  (Signal 3)
        risk_regime_modifier: Optional regime filter [0.82, 1.15]  (Signal 4)

    Returns:
        Array of multipliers centred around 1.0. Values > 1 increase DCA
        weight for that day; values < 1 decrease it.
    """
    n = len(mvrv_zscore)

    # ── Defaults for optional parameters ─────────────────────────────────────
    if mvrv_acceleration is None:
        mvrv_acceleration = np.zeros(n)
    if mvrv_volatility is None:
        mvrv_volatility = np.full(n, 0.5)
    if signal_confidence is None:
        signal_confidence = np.full(n, 0.5)
    #if polymarket_sentiment is None:
        #polymarket_sentiment = np.full(n, 0.5)
    if churn_signal is None:
        churn_signal = np.zeros(n)
    if macro_modifier is None:
        macro_modifier = np.ones(n)
    if whale_signal is None:
        whale_signal = np.zeros(n)
    if risk_regime_modifier is None:
        risk_regime_modifier = np.ones(n)

    # ── 1. MVRV value signal (Signal 4 regime filter applied here) ────────────
    value_signal = -mvrv_zscore
    extreme_boost = compute_asymmetric_extreme_boost(mvrv_zscore)
    value_signal = (value_signal + extreme_boost) * risk_regime_modifier
    # risk_regime_modifier > 1 in bear regime (amplify buy), < 1 in bull regime (dampen)

    # ── 2. MA signal with adaptive trend modulation ───────────────────────────
    ma_signal = -price_vs_ma
    trend_modifier = compute_adaptive_trend_modifier(mvrv_gradient, mvrv_zscore)
    ma_signal = ma_signal * trend_modifier

    # ── 3. PM BTC sentiment (re-centred from [0,1] to [-0.1, 0.1]) ───────────
    #polymarket_signal = (polymarket_sentiment - 0.5) * 0.2

    # ── 4. Active churn signal (Signal 1) ────────────────────────────────────
    # Already in approximately [-2, 2]; scale to a reasonable contribution range
    # Positive churn (net outflow + active addrs) = buy more
    churn_contrib = np.tanh(churn_signal * 0.7)  # Soft-clip to ~[-0.6, 0.6]

    # ── 5. Whale signal (Signal 3) ────────────────────────────────────────────
    # z-score already in [-2, 2]; scale similarly
    whale_contrib = np.tanh(whale_signal * 0.5)  # Soft-clip to ~[-0.46, 0.46]

    # ── Weighted combination ──────────────────────────────────────────────────
    combined = (
        value_signal      * W_MVRV           # 40%
        + ma_signal       * W_MA             # 12%
        #+ polymarket_signal * W_PM_SENTIMENT  # 8%
        + churn_contrib   * W_CHURN          # 18%
        + whale_contrib   * W_WHALE          # 10%
    )

    # ── Acceleration modifier (from example_1, subtle: [0.85, 1.15]) ─────────
    accel_modifier = compute_acceleration_modifier(mvrv_acceleration, mvrv_gradient)
    accel_subtle = np.clip(0.85 + 0.30 * (accel_modifier - 0.5) / 0.5, 0.85, 1.15)
    combined = combined * accel_subtle

    # ── Confidence boost (from example_1) ────────────────────────────────────
    confidence_boost = np.where(
        signal_confidence > 0.7,
        1.0 + 0.15 * (signal_confidence - 0.7) / 0.3,
        1.0,
    )
    combined = combined * confidence_boost

    # ── MVRV volatility dampening (from example_1) ───────────────────────────
    volatility_dampening = np.where(
        mvrv_volatility > 0.8,
        1.0 - MVRV_VOLATILITY_DAMPENING * (mvrv_volatility - 0.8) / 0.2,
        1.0,
    )
    combined = combined * volatility_dampening

    # ── Macro event modifier (Signal 2 — applied last) ────────────────────────
    # This is the outermost gate: reduces position sizing near event resolutions
    # and during "no active event" regimes where downside risk is elevated.
    combined = combined * macro_modifier

    # ── Exponentiate to multiplier ────────────────────────────────────────────
    adjustment = combined * DYNAMIC_STRENGTH
    adjustment = np.clip(adjustment, -5, 100)
    multiplier = np.exp(adjustment)
    return np.where(np.isfinite(multiplier), multiplier, 1.0)


# =============================================================================
# Weight Computation API  
# =============================================================================


def compute_weights_fast(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n_past: int | None = None,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Compute weights for a date window using precomputed features.

    Drop-in replacement for example_1's compute_weights_fast().
    Reads new feature columns when present; falls back to defaults if absent.

    Args:
        features_df: DataFrame from precompute_features().
        start_date: Window start.
        end_date: Window end.
        n_past: Number of past/current days (locked).
        locked_weights: Optional pre-computed locked weights from database.

    Returns:
        Series of weights indexed by date.
    """
    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    base = np.ones(n) / n

    def _get(col, default):
        if col in df.columns:
            arr = _clean_array(df[col].values)
            return arr
        return np.full(n, default)

    price_vs_ma      = _get("price_vs_ma", 0.0)
    mvrv_zscore      = _get("mvrv_zscore", 0.0)
    mvrv_gradient    = _get("mvrv_gradient", 0.0)
    mvrv_acceleration = _get("mvrv_acceleration", 0.0)
    mvrv_volatility  = np.where(
        _get("mvrv_volatility", 0.5) == 0, 0.5, _get("mvrv_volatility", 0.5)
    )
    signal_confidence = np.where(
        _get("signal_confidence", 0.5) == 0, 0.5, _get("signal_confidence", 0.5)
    )
    # polymarket_sentiment = np.where(
    #     _get("polymarket_sentiment", 0.5) == 0, 0.5, _get("polymarket_sentiment", 0.5)
    # )
    churn_signal         = _get("churn_signal", 0.0)
    macro_modifier       = np.where(
        _get("macro_modifier", 1.0) == 0, 1.0, _get("macro_modifier", 1.0)
    )
    whale_signal         = _get("whale_signal", 0.0)
    risk_regime_modifier = np.where(
        _get("risk_regime_modifier", 1.0) == 0, 1.0, _get("risk_regime_modifier", 1.0)
    )

    dyn = compute_dynamic_multiplier(
        price_vs_ma=price_vs_ma,
        mvrv_zscore=mvrv_zscore,
        mvrv_gradient=mvrv_gradient,
        mvrv_acceleration=mvrv_acceleration,
        mvrv_volatility=mvrv_volatility,
        signal_confidence=signal_confidence,
        #polymarket_sentiment=polymarket_sentiment,
        churn_signal=churn_signal,
        macro_modifier=macro_modifier,
        whale_signal=whale_signal,
        risk_regime_modifier=risk_regime_modifier,
    )
    raw = base * dyn

    if n_past is None:
        n_past = n
    weights = allocate_sequential_stable(raw, n_past, locked_weights)
    return pd.Series(weights, index=df.index)


def compute_window_weights(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Compute weights for a date range with lock-on-compute stability.

    Identical interface to example_1 — drop-in replacement.

    Args:
        features_df: DataFrame from precompute_features().
        start_date: Investment window start.
        end_date: Investment window end.
        current_date: Current date (past/future boundary).
        locked_weights: Optional locked weights from database.

    Returns:
        Series of weights summing to 1.0.
    """
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")

    missing = full_range.difference(features_df.index)
    if len(missing) > 0:
        placeholder = pd.DataFrame(
            {col: 0.0 for col in features_df.columns},
            index=missing,
        )
        # Safe defaults for non-zero neutral values
        for col, val in [
            ("mvrv_zone", 0),
            ("mvrv_volatility", 0.5),
            ("signal_confidence", 0.5),
            #("polymarket_sentiment", 0.5),
            ("macro_modifier", 1.0),
            ("risk_regime_modifier", 1.0),
        ]:
            if col in placeholder.columns:
                placeholder[col] = val
        features_df = pd.concat([features_df, placeholder]).sort_index()

    past_end = min(current_date, end_date)
    if start_date <= past_end:
        n_past = len(pd.date_range(start=start_date, end=past_end, freq="D"))
    else:
        n_past = 0

    weights = compute_weights_fast(
        features_df, start_date, end_date, n_past, locked_weights
    )
    return weights.reindex(full_range, fill_value=0.0)


# =============================================================================
# Internal Utility
# =============================================================================


def _to_pandas(obj) -> pd.DataFrame | None:
    """Convert a Polars DataFrame to pandas, or pass through if already pandas."""
    if obj is None:
        return None
    try:
        import polars as pl
        if isinstance(obj, pl.DataFrame):
            return obj.to_pandas()
    except ImportError:
        pass
    if isinstance(obj, pd.DataFrame):
        return obj
    return None
