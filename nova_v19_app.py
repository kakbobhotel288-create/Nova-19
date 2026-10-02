# ================================================================
# NOVA V19 — POCKET OPTION DEMO STREAMLIT APP
# ================================================================
#
# Run:
#   pip install streamlit yfinance scikit-learn pandas numpy
#   streamlit run nova_v19_app.py
#
# IMPORTANT:
# - EUR/USD only
# - 5-minute candles
# - 5-minute expiry
# - Manual demo execution ONLY
# - No automatic Pocket Option orders
# - No martingale
# ================================================================

import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ================================================================
# PAGE
# ================================================================

st.set_page_config(
    page_title="NOVA V19 Demo",
    page_icon="📈",
    layout="wide",
)


# ================================================================
# CONFIG
# ================================================================

TICKER = "EURUSD=X"
INTERVAL = "5m"
PERIOD = "60d"

SIGNAL_THRESHOLD = 0.60
PAYOUT = 0.92
TRAIN_HOLDOUT = 0.15

# Mountain Time / Utah
LOCAL_TZ = ZoneInfo("America/Denver")

SIGNAL_LOG_FILE = "nova_v19_demo_signals.csv"
TRADE_LOG_FILE = "nova_v19_demo_trades.csv"


# ================================================================
# V19 FEATURES — EXACT 34
# ================================================================

FEATURES = [
    "ret_1",
    "ret_2",
    "ret_3",
    "ret_6",
    "ret_12",
    "ret_24",
    "ret_48",

    "body_ratio",
    "signed_body",
    "upper_wick_ratio",
    "lower_wick_ratio",

    "atr_ratio",

    "ema9_dist",
    "ema21_dist",
    "ema50_dist",

    "ema9_slope",
    "ema21_slope",
    "ema50_slope",

    "trend_strength",

    "rsi",
    "rsi_centered",

    "momentum_3",
    "momentum_6",
    "momentum_12",

    "range_pos_6",
    "range_pos_12",
    "range_pos_24",
    "range_pos_48",

    "hour_sin",
    "hour_cos",

    "session_asia",
    "session_europe",
    "session_newyork",
    "session_evening",
]

assert len(FEATURES) == 34


# ================================================================
# CSV COLUMNS
# ================================================================

SIGNAL_COLUMNS = [
    "signal_id",
    "candle_time_utc",
    "candle_time_local",
    "price",
    "p_up",
    "p_down",
    "signal",
    "expiry_time_utc",
    "expiry_time_local",
]

TRADE_COLUMNS = [
    "trade_id",
    "signal_time_utc",
    "signal_time_local",
    "entry_time_utc",
    "entry_time_local",
    "entry_price",
    "p_up",
    "signal",
    "expiry_time_utc",
    "expiry_time_local",
    "expiry_price",
    "result",
    "demo_return",
    "notes",
]


# ================================================================
# TITLE
# ================================================================

st.title("📈 NOVA V19")

st.caption(
    "EUR/USD • 5-minute model • 5-minute expiry • "
    "Pocket Option demo • Manual execution only"
)


# ================================================================
# DATA DOWNLOAD
# ================================================================

@st.cache_data(
    ttl=20,
    show_spinner=False
)
def download_data():

    raw = yf.download(
        TICKER,
        interval=INTERVAL,
        period=PERIOD,
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if raw is None or len(raw) == 0:
        raise RuntimeError(
            "Yahoo Finance returned no data."
        )

    # Handle Yahoo MultiIndex
    if isinstance(raw.columns, pd.MultiIndex):

        raw.columns = [
            c[0] if isinstance(c, tuple) else c
            for c in raw.columns
        ]

    raw.columns = [
        str(c).lower()
        for c in raw.columns
    ]

    raw = raw.loc[
        :,
        ~raw.columns.duplicated()
    ]

    required = [
        "open",
        "high",
        "low",
        "close"
    ]

    missing = [
        c for c in required
        if c not in raw.columns
    ]

    if missing:
        raise KeyError(
            f"Missing columns: {missing}"
        )

    data = raw[required].copy()

    for c in required:

        data[c] = pd.to_numeric(
            data[c],
            errors="coerce"
        )

    data = data.dropna()

    # Convert index to UTC
    if data.index.tz is None:

        data.index = data.index.tz_localize(
            "UTC"
        )

    else:

        data.index = data.index.tz_convert(
            "UTC"
        )

    data = data.sort_index()

    data = data[
        ~data.index.duplicated(
            keep="last"
        )
    ]

    return data


# ================================================================
# FEATURE ENGINEERING
# ================================================================

def build_features(df):

    x = df.copy()

    close = x["close"]
    open_ = x["open"]
    high = x["high"]
    low = x["low"]

    eps = 1e-12

    # ------------------------------------------------------------
    # RETURNS
    # ------------------------------------------------------------

    x["ret_1"] = close.pct_change(1)
    x["ret_2"] = close.pct_change(2)
    x["ret_3"] = close.pct_change(3)
    x["ret_6"] = close.pct_change(6)
    x["ret_12"] = close.pct_change(12)
    x["ret_24"] = close.pct_change(24)
    x["ret_48"] = close.pct_change(48)

    # ------------------------------------------------------------
    # CANDLE STRUCTURE
    # ------------------------------------------------------------

    candle_range = (
        high - low
    ).replace(
        0,
        np.nan
    )

    body = close - open_

    x["body_ratio"] = (
        body.abs() /
        (candle_range + eps)
    )

    x["signed_body"] = (
        body /
        (candle_range + eps)
    )

    x["upper_wick_ratio"] = (
        high -
        np.maximum(
            open_,
            close
        )
    ) / (
        candle_range + eps
    )

    x["lower_wick_ratio"] = (
        np.minimum(
            open_,
            close
        ) - low
    ) / (
        candle_range + eps
    )

    # ------------------------------------------------------------
    # ATR
    # ------------------------------------------------------------

    previous_close = close.shift(1)

    tr1 = high - low

    tr2 = (
        high -
        previous_close
    ).abs()

    tr3 = (
        low -
        previous_close
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(axis=1)

    atr = true_range.rolling(
        14
    ).mean()

    x["atr_ratio"] = (
        atr /
        (close + eps)
    )

    # ------------------------------------------------------------
    # EMAs
    # ------------------------------------------------------------

    ema9 = close.ewm(
        span=9,
        adjust=False
    ).mean()

    ema21 = close.ewm(
        span=21,
        adjust=False
    ).mean()

    ema50 = close.ewm(
        span=50,
        adjust=False
    ).mean()

    x["ema9_dist"] = (
        close - ema9
    ) / (
        close + eps
    )

    x["ema21_dist"] = (
        close - ema21
    ) / (
        close + eps
    )

    x["ema50_dist"] = (
        close - ema50
    ) / (
        close + eps
    )

    x["ema9_slope"] = (
        ema9.pct_change(3)
    )

    x["ema21_slope"] = (
        ema21.pct_change(3)
    )

    x["ema50_slope"] = (
        ema50.pct_change(3)
    )

    x["trend_strength"] = (
        (ema9 - ema21) +
        (ema21 - ema50)
    ) / (
        close + eps
    )

    # ------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.rolling(
        14
    ).mean()

    avg_loss = loss.rolling(
        14
    ).mean()

    rs = (
        avg_gain /
        (avg_loss + eps)
    )

    rsi = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    x["rsi"] = rsi

    x["rsi_centered"] = (
        rsi - 50
    ) / 50

    # ------------------------------------------------------------
    # MOMENTUM
    # ------------------------------------------------------------

    x["momentum_3"] = (
        close.pct_change(3)
    )

    x["momentum_6"] = (
        close.pct_change(6)
    )

    x["momentum_12"] = (
        close.pct_change(12)
    )

    # ------------------------------------------------------------
    # RANGE POSITION
    # ------------------------------------------------------------

    for n in [
        6,
        12,
        24,
        48
    ]:

        rolling_high = (
            high.rolling(n).max()
        )

        rolling_low = (
            low.rolling(n).min()
        )

        x[
            f"range_pos_{n}"
        ] = (
            close - rolling_low
        ) / (
            rolling_high -
            rolling_low +
            eps
        )

    # ------------------------------------------------------------
    # TIME FEATURES
    # ------------------------------------------------------------

    hour = x.index.hour

    x["hour_sin"] = np.sin(
        2 *
        np.pi *
        hour /
        24
    )

    x["hour_cos"] = np.cos(
        2 *
        np.pi *
        hour /
        24
    )

    # ------------------------------------------------------------
    # SESSIONS — UTC
    # ------------------------------------------------------------

    x["session_asia"] = (
        (
            (hour >= 0) &
            (hour < 8)
        ).astype(int)
    )

    x["session_europe"] = (
        (
            (hour >= 7) &
            (hour < 16)
        ).astype(int)
    )

    x["session_newyork"] = (
        (
            (hour >= 13) &
            (hour < 21)
        ).astype(int)
    )

    x["session_evening"] = (
        (
            (hour >= 21) |
            (hour < 1)
        ).astype(int)
    )

    return x


# ================================================================
# TRAIN V19
# ================================================================

@st.cache_resource(
    show_spinner=True
)
def train_model():

    data = download_data()

    features = build_features(
        data
    )

    # ------------------------------------------------------------
    # TARGET
    # ------------------------------------------------------------

    next_close = (
        data["close"].shift(-1)
    )

    features["target"] = np.where(
        next_close >
        data["close"],
        1,
        np.where(
            next_close <
            data["close"],
            0,
            np.nan
        )
    )

    # Target endpoint
    features["target_end"] = pd.Series(
        data.index.to_series().shift(-1),
        index=data.index
    )

    model_df = features[
        FEATURES +
        [
            "target",
            "target_end"
        ]
    ].dropna()

    # Only UP/DOWN
    model_df = model_df[
        model_df["target"].isin(
            [0, 1]
        )
    ].copy()

    model_df["target"] = (
        model_df["target"]
        .astype(int)
    )

    # ------------------------------------------------------------
    # CHRONOLOGICAL SPLIT
    # ------------------------------------------------------------

    split_idx = int(
        len(model_df) *
        (
            1 -
            TRAIN_HOLDOUT
        )
    )

    holdout_start = (
        model_df.index[
            split_idx
        ]
    )

    # TRUE PURGE
    train_df = model_df[
        model_df["target_end"] <
        holdout_start
    ].copy()

    # ------------------------------------------------------------
    # V19 LOGISTIC MODEL
    # ------------------------------------------------------------

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),

            (
                "classifier",
                LogisticRegression(
                    C=0.10,
                    max_iter=3000,
                    solver="lbfgs",
                    random_state=42
                )
            )
        ]
    )

    model.fit(
        train_df[FEATURES],
        train_df["target"]
    )

    return model


# ================================================================
# GET CURRENT NOVA SIGNAL
# ================================================================

@st.cache_data(
    ttl=20,
    show_spinner=False
)
def get_latest_signal():

    data = download_data()

    model = train_model()

    feat = build_features(
        data
    )

    # ------------------------------------------------------------
    # IMPORTANT
    #
    # Yahoo may contain the currently-forming candle.
    # Therefore:
    #
    # - [-1] = potentially forming
    # - [-2] = latest completed candle
    # ------------------------------------------------------------

    if len(feat) < 3:

        raise RuntimeError(
            "Not enough candles."
        )

    completed_time = (
        feat.index[-2]
    )

    row = feat.loc[
        completed_time
    ]

    if row[
        FEATURES
    ].isna().any():

        missing = row[
            FEATURES
        ].index[
            row[
                FEATURES
            ].isna()
        ].tolist()

        raise RuntimeError(
            "Missing features: "
            +
            str(missing)
        )

    X = pd.DataFrame(
        [
            row[
                FEATURES
            ].values
        ],
        columns=FEATURES
    )

    probabilities = (
        model.predict_proba(X)[0]
    )

    classes = (
        model.classes_
    )

    p_down = float(
        probabilities[
            list(classes).index(0)
        ]
    )

    p_up = float(
        probabilities[
            list(classes).index(1)
        ]
    )

    # ------------------------------------------------------------
    # V19 DECISION
    # ------------------------------------------------------------

    if p_up >= SIGNAL_THRESHOLD:

        signal = "CALL"

    elif p_down >= SIGNAL_THRESHOLD:

        signal = "PUT"

    else:

        signal = "WAIT"

    # ------------------------------------------------------------
    # EXPIRY
    # ------------------------------------------------------------

    expiry_time = (
        completed_time +
        timedelta(
            minutes=5
        )
    )

    return {
        "candle_time":
            completed_time,

        "price":
            float(
                data.loc[
                    completed_time,
                    "close"
                ]
            ),

        "p_up":
            p_up,

        "p_down":
            p_down,

        "signal":
            signal,

        "expiry_time":
            expiry_time
    }


# ================================================================
# CSV LOADING
# ================================================================

def load_csv(
    filename,
    columns
):

    if os.path.exists(
        filename
    ):

        try:

            df = pd.read_csv(
                filename
            )

            for c in columns:

                if c not in df.columns:

                    df[c] = ""

            return df

        except Exception:

            pass

    return pd.DataFrame(
        columns=columns
    )


# ================================================================
# SAVE SIGNAL
# ================================================================

def save_signal(
    sig
):

    df = load_csv(
        SIGNAL_LOG_FILE,
        SIGNAL_COLUMNS
    )

    candle_string = (
        sig["candle_time"]
        .isoformat()
    )

    # Prevent duplicate candles
    if len(df) > 0:

        existing = (
            df[
                "candle_time_utc"
            ]
            .astype(str)
        )

        if (
            candle_string
            in
            existing.values
        ):

            return

    signal_id = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%d%H%M%S"
        )
    )

    row = {

        "signal_id":
            signal_id,

        "candle_time_utc":
            candle_string,

        "candle_time_local":
            sig[
                "candle_time"
            ]
            .astimezone(
                LOCAL_TZ
            )
            .isoformat(),

        "price":
            sig["price"],

        "p_up":
            sig["p_up"],

        "p_down":
            sig["p_down"],

        "signal":
            sig["signal"],

        "expiry_time_utc":
            sig[
                "expiry_time"
            ]
            .isoformat(),

        "expiry_time_local":
            sig[
                "expiry_time"
            ]
            .astimezone(
                LOCAL_TZ
            )
            .isoformat()
    }

    result = pd.concat(
        [
            df,
            pd.DataFrame(
                [row]
            )
        ],
        ignore_index=True
    )

    result.to_csv(
        SIGNAL_LOG_FILE,
        index=False
    )


# ================================================================
# PERFORMANCE
# ================================================================

def get_performance():

    df = load_csv(
        TRADE_LOG_FILE,
        TRADE_COLUMNS
    )

    if len(df) == 0:

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "win_rate": 0,
            "net": 0,
            "call_rate": 0,
            "put_rate": 0
        }

    results = (
        df["result"]
        .astype(str)
        .str.upper()
    )

    wins = int(
        (results == "WIN")
        .sum()
    )

    losses = int(
        (results == "LOSS")
        .sum()
    )

    pushes = int(
        (results == "PUSH")
        .sum()
    )

    settled = (
        wins +
        losses
    )

    win_rate = (
        wins /
        settled *
        100
        if settled > 0
        else 0
    )

    net = pd.to_numeric(
        df["demo_return"],
        errors="coerce"
    ).fillna(0).sum()

    # ------------------------------------------------------------
    # CALL
    # ------------------------------------------------------------

    call = df[
        df[
            "signal"
        ]
        .astype(str)
        .str.upper()
        == "CALL"
    ]

    if len(call) > 0:

        call_results = (
            call["result"]
            .astype(str)
            .str.upper()
        )

        call_wins = int(
            (
                call_results ==
                "WIN"
            ).sum()
        )

        call_losses = int(
            (
                call_results ==
                "LOSS"
            ).sum()
        )

        call_settled = (
            call_wins +
            call_losses
        )

        call_rate = (
            call_wins /
            call_settled *
            100
            if call_settled
            else 0
        )

    else:

        call_rate = 0

    # ------------------------------------------------------------
    # PUT
    # ------------------------------------------------------------

    put = df[
        df[
            "signal"
        ]
        .astype(str)
        .str.upper()
        == "PUT"
    ]

    if len(put) > 0:

        put_results = (
            put["result"]
            .astype(str)
            .str.upper()
        )

        put_wins = int(
            (
                put_results ==
                "WIN"
            ).sum()
        )

        put_losses = int(
            (
                put_results ==
                "LOSS"
            ).sum()
        )

        put_settled = (
            put_wins +
            put_losses
        )

        put_rate = (
            put_wins /
            put_settled *
            100
            if put_settled
            else 0
        )

    else:

        put_rate = 0

    return {

        "trades":
            len(df),

        "wins":
            wins,

        "losses":
            losses,

        "pushes":
            pushes,

        "win_rate":
            win_rate,

        "net":
            net,

        "call_rate":
            call_rate,

        "put_rate":
            put_rate
    }


# ================================================================
# SAVE TRADE
# ================================================================

def save_trade(
    signal_time,
    entry_time,
    entry_price,
    p_up,
    signal,
    expiry_time,
    expiry_price,
    result,
    notes
):

    df = load_csv(
        TRADE_LOG_FILE,
        TRADE_COLUMNS
    )

    result = result.upper()

    if result == "WIN":

        demo_return = PAYOUT

    elif result == "LOSS":

        demo_return = -1.0

    elif result == "PUSH":

        demo_return = 0.0

    else:

        raise ValueError(
            "Invalid result."
        )

    signal_timestamp = (
        pd.Timestamp(
            signal_time
        )
    )

    entry_timestamp = (
        pd.Timestamp(
            entry_time
        )
    )

    expiry_timestamp = (
        pd.Timestamp(
            expiry_time
        )
    )

    # Ensure UTC
    if (
        signal_timestamp.tzinfo
        is None
    ):

        signal_timestamp = (
            signal_timestamp
            .tz_localize("UTC")
        )

    else:

        signal_timestamp = (
            signal_timestamp
            .tz_convert("UTC")
        )

    if (
        entry_timestamp.tzinfo
        is None
    ):

        entry_timestamp = (
            entry_timestamp
            .tz_localize("UTC")
        )

    else:

        entry_timestamp = (
            entry_timestamp
            .tz_convert("UTC")
        )

    if (
        expiry_timestamp.tzinfo
        is None
    ):

        expiry_timestamp = (
            expiry_timestamp
            .tz_localize("UTC")
        )

    else:

        expiry_timestamp = (
            expiry_timestamp
            .tz_convert("UTC")
        )

    trade_id = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%d%H%M%S%f"
        )
    )

    row = {

        "trade_id":
            trade_id,

        "signal_time_utc":
            signal_timestamp.isoformat(),

        "signal_time_local":
            signal_timestamp
            .tz_convert(
                LOCAL_TZ
            )
            .isoformat(),

        "entry_time_utc":
            entry_timestamp.isoformat(),

        "entry_time_local":
            entry_timestamp
            .tz_convert(
                LOCAL_TZ
            )
            .isoformat(),

        "entry_price":
            entry_price,

        "p_up":
            p_up,

        "signal":
            signal,

        "expiry_time_utc":
            expiry_timestamp.isoformat(),

        "expiry_time_local":
            expiry_timestamp
            .tz_convert(
                LOCAL_TZ
            )
            .isoformat(),

        "expiry_price":
            expiry_price,

        "result":
            result,

        "demo_return":
            demo_return,

        "notes":
            notes
    }

    result_df = pd.concat(
        [
            df,
            pd.DataFrame(
                [row]
            )
        ],
        ignore_index=True
    )

    result_df.to_csv(
        TRADE_LOG_FILE,
        index=False
    )


# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:

    st.header("⚙️ NOVA Controls")

    st.write(
        "**Asset:** EUR/USD"
    )

    st.write(
        "**Chart:** 5-minute"
    )

    st.write(
        "**Expiry:** 5 minutes"
    )

    st.write(
        f"**Threshold:** {SIGNAL_THRESHOLD:.2f}"
    )

    st.write(
        f"**Payout assumption:** {PAYOUT:.2f}"
    )

    st.divider()

    if st.button(
        "🔄 Refresh data",
        use_container_width=True
    ):

        st.cache_data.clear()

        st.rerun()

    st.divider()

    st.warning(
        "DEMO ONLY\n\n"
        "NOVA does not automatically place "
        "Pocket Option orders."
    )


# ================================================================
# GET SIGNAL
# ================================================================

try:

    signal = get_latest_signal()

    save_signal(
        signal
    )

except Exception as e:

    st.error(
        f"Data/model error: {e}"
    )

    st.stop()


# ================================================================
# LIVE CLOCK
# ================================================================

now_local = datetime.now(
    LOCAL_TZ
)

now_utc = datetime.now(
    timezone.utc
)

st.subheader("🕐 Live Clock")

clock_col1, clock_col2 = st.columns(2)

clock_col1.metric(
    "Local time",
    now_local.strftime(
        "%I:%M:%S %p"
    )
)

clock_col2.metric(
    "UTC time",
    now_utc.strftime(
        "%H:%M:%S"
    )
)

st.caption(
    now_local.strftime(
        "%A, %B %d, %Y — %Z"
    )
)


# ================================================================
# SIGNAL
# ================================================================

st.divider()

st.subheader(
    "🎯 Current NOVA V19 Signal"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "EUR/USD",
    f"{signal['price']:.6f}"
)

col2.metric(
    "P(UP)",
    f"{signal['p_up'] * 100:.2f}%"
)

col3.metric(
    "P(DOWN)",
    f"{signal['p_down'] * 100:.2f}%"
)

col4.metric(
    "Signal",
    signal["signal"]
)


# ================================================================
# SIGNAL STATUS
# ================================================================

if signal["signal"] == "CALL":

    st.success(
        "🟢 CALL SIGNAL — "
        "P(UP) reached the 0.60 threshold."
    )

elif signal["signal"] == "PUT":

    st.error(
        "🔴 PUT SIGNAL — "
        "P(DOWN) reached the 0.60 threshold."
    )

else:

    st.info(
        "⚪ WAIT — "
        "Neither direction reached the 0.60 threshold."
    )


# ================================================================
# CANDLE INFORMATION
# ================================================================

st.subheader(
    "🕯️ Candle Information"
)

c1, c2, c3 = st.columns(3)

local_candle = (
    signal["candle_time"]
    .astimezone(
        LOCAL_TZ
    )
)

local_expiry = (
    signal["expiry_time"]
    .astimezone(
        LOCAL_TZ
    )
)

c1.write(
    "**Latest completed candle**"
)

c1.write(
    local_candle.strftime(
        "%Y-%m-%d %I:%M:%S %p %Z"
    )
)

c2.write(
    "**Signal expiry**"
)

c2.write(
    local_expiry.strftime(
        "%Y-%m-%d %I:%M:%S %p %Z"
    )
)

c3.write(
    "**Model source**"
)

c3.write(
    "Yahoo Finance EUR/USD"
)


# ================================================================
# PERFORMANCE
# ================================================================

st.divider()

st.subheader(
    "📊 Demo Performance"
)

performance = get_performance()

p1, p2, p3, p4, p5, p6 = st.columns(6)

p1.metric(
    "Trades",
    performance["trades"]
)

p2.metric(
    "Wins",
    performance["wins"]
)

p3.metric(
    "Losses",
    performance["losses"]
)

p4.metric(
    "Win rate",
    f"{performance['win_rate']:.2f}%"
)

p5.metric(
    "CALL rate",
    f"{performance['call_rate']:.2f}%"
)

p6.metric(
    "PUT rate",
    f"{performance['put_rate']:.2f}%"
)

st.metric(
    "Demo net return",
    f"{performance['net']:.2f} units"
)

st.caption(
    f"Model payout assumption: {PAYOUT:.2f}x. "
    f"Break-even under this assumption: "
    f"{100 / (1 + PAYOUT):.2f}%."
)


# ================================================================
# RECORD TRADE
# ================================================================

st.divider()

st.subheader(
    "📝 Record Demo Trade"
)

st.write(
    "After the 5-minute expiration, enter the actual "
    "Pocket Option prices and result here."
)

with st.form(
    "trade_form"
):

    left, right = st.columns(2)

    with left:

        signal_time = st.text_input(
            "Signal time (UTC)",
            value=signal[
                "candle_time"
            ].strftime(
                "%Y-%m-%d %H:%M:%S+00:00"
            )
        )

        entry_time = st.text_input(
            "Entry time (UTC)",
            value=datetime.now(
                timezone.utc
            ).strftime(
                "%Y-%m-%d %H:%M:%S+00:00"
            )
        )

        entry_price = st.number_input(
            "Pocket Option entry price",
            min_value=0.0,
            value=float(
                signal["price"]
            ),
            format="%.6f"
        )

        selected_signal = st.selectbox(
            "Signal",
            [
                "CALL",
                "PUT"
            ],
            index=(
                0
                if signal["signal"]
                == "CALL"
                else 1
            )
        )

    with right:

        p_up = st.number_input(
            "NOVA P(UP)",
            min_value=0.0,
            max_value=1.0,
            value=float(
                signal["p_up"]
            ),
            step=0.01
        )

        expiry_time = st.text_input(
            "Expiry time (UTC)",
            value=signal[
                "expiry_time"
            ].strftime(
                "%Y-%m-%d %H:%M:%S+00:00"
            )
        )

        expiry_price = st.number_input(
            "Pocket Option expiry price",
            min_value=0.0,
            value=float(
                signal["price"]
            ),
            format="%.6f"
        )

        result = st.selectbox(
            "Result",
            [
                "WIN",
                "LOSS",
                "PUSH"
            ]
        )

    notes = st.text_input(
        "Notes"
    )

    submitted = st.form_submit_button(
        "💾 Save Demo Trade",
        use_container_width=True
    )

    if submitted:

        try:

            save_trade(
                signal_time=
                    signal_time,

                entry_time=
                    entry_time,

                entry_price=
                    entry_price,

                p_up=
                    p_up,

                signal=
                    selected_signal,

                expiry_time=
                    expiry_time,

                expiry_price=
                    expiry_price,

                result=
                    result,

                notes=
                    notes
            )

            st.success(
                "Demo trade saved successfully."
            )

            st.rerun()

        except Exception as e:

            st.error(
                f"Could not save trade: {e}"
            )


# ================================================================
# RECENT TRADES
# ================================================================

st.divider()

st.subheader(
    "📋 Recent Demo Trades"
)

trades = load_csv(
    TRADE_LOG_FILE,
    TRADE_COLUMNS
)

if len(trades) > 0:

    display_columns = [
        "signal_time_local",
        "signal",
        "p_up",
        "entry_price",
        "expiry_price",
        "result",
        "demo_return",
        "notes"
    ]

    st.dataframe(
        trades.tail(25)[
            display_columns
        ],
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No completed demo trades yet."
    )


# ================================================================
# RECENT SIGNALS
# ================================================================

st.subheader(
    "📡 Recent NOVA Signals"
)

signals = load_csv(
    SIGNAL_LOG_FILE,
    SIGNAL_COLUMNS
)

if len(signals) > 0:

    display_columns = [
        "candle_time_local",
        "price",
        "p_up",
        "p_down",
        "signal",
        "expiry_time_local"
    ]

    st.dataframe(
        signals.tail(25)[
            display_columns
        ],
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No signals logged yet."
    )


# ================================================================
# FOOTER
# ================================================================

st.divider()

st.caption(
    "NOVA V19 • EUR/USD • 5m • Demo/Paper Testing • "
    "Manual execution only"
)

st.caption(
    "The displayed Yahoo Finance price may differ from the "
    "price shown by Pocket Option. Verify the platform price "
    "before manually recording a demo trade."
)


# ================================================================
# AUTO REFRESH
# ================================================================

# Streamlit reruns the entire app after this interval.
# This updates the live clock and checks for a new 5-minute candle.

st.markdown(
    """
    <script>
        setTimeout(function() {
            window.parent.location.reload();
        }, 10000);
    </script>
    """,
    unsafe_allow_html=True
)
