#!/usr/bin/env python3
"""
hedger_ml_live.py  —  ML-driven Δ-hedger   (NaN-safe, coloured buy / sell markers)
"""
import json, joblib, sqlite3, warnings
from pathlib   import Path
from datetime  import timedelta
import numpy   as np
import pandas  as pd
import matplotlib.pyplot as plt
import matplotlib.dates  as mdates
from matplotlib.ticker   import MaxNLocator
from scipy.stats         import norm
from scipy.optimize      import brentq

# ─── User parameters ──────────────────────────────────────────────────
STRIKE        = 620
ML_THRESHOLD  = 0.9
MAX_TRADES    = 100
COOLDOWN_MIN  = 10
MAKE_PLOT     = True

EXPIRY     = pd.Timestamp("2025-05-16")
RISK_FREE  = 0.06

CALL_DB = Path("META_call.db")
PUT_DB  = Path("META_put.db")
TABLE_C = f"META2516E{STRIKE}"
TABLE_P = f"META2516Q{STRIKE}"

# ─── Load trained model & metadata ────────────────────────────────────
model = joblib.load("hedge_model.pkl")
with open("hedge_meta.json") as f:
    meta = json.load(f)
feat_cols = meta["features"]

# ─── Black-Scholes helpers ────────────────────────────────────────────
def bs_d1(S,K,tau,r,sigma):
    return (np.log(S/K)+(r+0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))

def bs_price(S,K,tau,r,sigma,opt="call"):
    d1=bs_d1(S,K,tau,r,sigma); d2=d1-sigma*np.sqrt(tau); disc=K*np.exp(-r*tau)
    return S*norm.cdf(d1)-disc*norm.cdf(d2) if opt=="call" \
           else disc*norm.cdf(-d2)-S*norm.cdf(-d1)

def iv_brent(S,K,tau,r,mkt,opt):
    try: return brentq(lambda s: bs_price(S,K,tau,r,s,opt)-mkt,1e-5,5.0)
    except ValueError: return np.nan

# ─── Load option tables ───────────────────────────────────────────────
with sqlite3.connect(CALL_DB) as con:
    dfC = (pd.read_sql(f"SELECT V_time,C_bid,C_ask,S_last FROM {TABLE_C}",
                       con, parse_dates=["V_time"])
             .sort_values("V_time")
             .rename(columns={"C_bid":"bid_C","C_ask":"ask_C"}))

with sqlite3.connect(PUT_DB) as con:
    dfP = (pd.read_sql(f"SELECT V_time,P_bid,P_ask,S_last FROM {TABLE_P}",
                       con, parse_dates=["V_time"])
             .sort_values("V_time")
             .rename(columns={"P_bid":"bid_P","P_ask":"ask_P"}))

df = (pd.merge(dfC, dfP, on="V_time", suffixes=("","_P"))
        .drop(columns=["S_last_P"]))

df["trade_date"] = df["V_time"].dt.normalize()
df["tau"]        = (EXPIRY - df["trade_date"]).dt.days/252
df = df[df.tau>0].reset_index(drop=True)

# ─── Greeks & feature columns ─────────────────────────────────────────
df["iv_C"] = df.apply(lambda r: iv_brent(r.S_last,STRIKE,r.tau,RISK_FREE,
                                         r.ask_C,"call"), axis=1)
df["iv_P"] = df.apply(lambda r: iv_brent(r.S_last,STRIKE,r.tau,RISK_FREE,
                                         r.ask_P,"put"),  axis=1)
d1C = bs_d1(df.S_last,STRIKE,df.tau,RISK_FREE,df.iv_C)
d1P = bs_d1(df.S_last,STRIKE,df.tau,RISK_FREE,df.iv_P)

df["delta_port"] = -norm.cdf(d1C) - (norm.cdf(d1P)-1)
df["delta_jump"] = df["delta_port"].diff().abs().fillna(0.0)
df["speed"]      = df.S_last.pct_change().fillna(0.0)
df["minutes_since_open"] = (df.V_time-df.V_time.iloc[0]).dt.total_seconds()/60
df["tick_index"] = np.arange(len(df))
df["strike"]     = STRIKE

# ─── Option P&L (short straddle: sell @ bid, buy back @ ask) ──────────
openC, openP = df.bid_C.iat[0], df.bid_P.iat[0]
df["pnl_opt"] = (openC-df.ask_C)+(openP-df.ask_P)

# ─── Live simulation ─────────────────────────────────────────────────
cash, stk_pos = 0.0, 0.0
last_trade    = df.V_time.iat[0]-timedelta(minutes=COOLDOWN_MIN+1)
trades        = 0
pnl_series    = []

buy_ts,  buy_pnl  = [], []   # ▲ markers
sell_ts, sell_pnl = [], []   # ▼ markers

for _, row in df.iterrows():
    X = np.array([[row[c] for c in feat_cols]], float)

    if np.isnan(X).any() or np.isinf(X).any():
        prob = 0.0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",category=UserWarning)
            prob = model.predict_proba(X)[0,1]

    cool_ok = (row.V_time-last_trade) >= timedelta(minutes=COOLDOWN_MIN)
    if prob>=ML_THRESHOLD and cool_ok and trades<MAX_TRADES:
        target = 0.8*row.delta_port
        d_pos  = target - stk_pos
        cash  -= d_pos * row.S_last
        stk_pos = target
        last_trade = row.V_time
        trades += 1

        hedge_val = cash + stk_pos*row.S_last + row.pnl_opt
        if d_pos > 0:    # we **bought** stock  (hedge more long)
            buy_ts.append(row.V_time);   buy_pnl.append(hedge_val)
        else:            # we **sold** stock   (hedge more short)
            sell_ts.append(row.V_time);  sell_pnl.append(hedge_val)

    # mark-to-market
    hedge_val = cash + stk_pos * row.S_last
    pnl_series.append(row.pnl_opt + hedge_val)

df["pnl_hedged"] = pnl_series

# ─── Summary ─────────────────────────────────────────────────────────
print("\n=== ML-driven hedge summary ===")
print(f"Trades executed          : {trades}/{MAX_TRADES}")
print(f"Final naked   P&L ($/str): {df.pnl_opt.iat[-1]*100: .2f}")
print(f"Final hedged  P&L ($/str): {df.pnl_hedged.iat[-1]*100: .2f}")
print("================================================\n")

# ─── Plot ─────────────────────────────────────────────────────────────
if MAKE_PLOT:
    fig, ax = plt.subplots(figsize=(15,6))
    ax.plot(df.V_time, df.pnl_opt,    lw=0.8, label="Naked P&L")
    ax.plot(df.V_time, df.pnl_hedged, lw=1.3, label="ML-hedged P&L")

    # green ▲ for buys, red ▼ for sells
    ax.scatter(buy_ts,  buy_pnl,  marker="^", color="green", zorder=6,
               label="Buy hedge (increase long)")
    ax.scatter(sell_ts, sell_pnl, marker="v", color="red",   zorder=6,
               label="Sell hedge (increase short)")

    ax2 = ax.twinx()
    ax2.plot(df.V_time, df.S_last, color="grey", alpha=0.25, label="Stock")
    ax2.set_ylabel("Spot price ($)")
    ax2.legend(loc="upper right", frameon=False)

    major = mdates.AutoDateLocator(minticks=6, maxticks=10)
    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(major))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune="both"))
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.grid(axis="x", ls=":",  alpha=0.3)
    ax.set_title(f"NVDA {STRIKE} straddle — ML-directed Δ-hedge")
    ax.set_ylabel("P&L per straddle ($)")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig("straddle_ml_hedge.png", dpi=300)
    plt.show()
