#!/usr/bin/env python3
"""
train_hedger_ml.py  (v4 – NaN-safe, JSON-safe)
==============================================

• Builds a tick-level dataset from strikes present in BOTH NVDA_call.db
  and NVDA_put.db, skipping any strike whose data all falls after EXPIRY.
• Drops rows that still contain NaN/inf in the feature set.
• Trains a StandardScaler + LogisticRegression model.
• Saves:
      hedge_model.pkl
      hedge_meta.json   (now JSON-serialisable)
"""

import json, sqlite3, joblib, warnings
from pathlib import Path
from datetime import timedelta
import numpy  as np
import pandas as pd
from scipy.stats     import norm
from scipy.optimize  import brentq
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import LogisticRegression
from sklearn.metrics       import classification_report

# ───────── CONFIG ─────────────────────────────────────────────────────
CALL_DB, PUT_DB = Path("NVDA_call.db"), Path("NVDA_put.db")
EXPIRY,  RISK_FREE = pd.Timestamp("2025-04-17"), 0.06

EPS_DELTA, EPS_JUMP = 0.25, 0.15
SPEED_UP,  SPEED_DN = 0.006, 0.006
COOLDOWN_MIN, N_TRAIN_STRIKES = 10, 8

# ───────── Black-Scholes helpers ──────────────────────────────────────
def bs_d1(S,K,tau,r,sigma):
    return (np.log(S/K)+(r+0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))

def bs_price(S,K,tau,r,sigma,opt="call"):
    d1=bs_d1(S,K,tau,r,sigma); d2=d1-sigma*np.sqrt(tau); disc=K*np.exp(-r*tau)
    return S*norm.cdf(d1)-disc*norm.cdf(d2) if opt=="call" \
           else disc*norm.cdf(-d2)-S*norm.cdf(-d1)

def iv_brent(S,K,tau,r,mkt,opt):
    try:  return brentq(lambda s: bs_price(S,K,tau,r,s,opt)-mkt,1e-5,5.0)
    except ValueError: return np.nan

# ───────── DB helpers ─────────────────────────────────────────────────
def tables_in(db):
    with sqlite3.connect(db) as con:
        return [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table';")]

def strike_from(t):           # NVDA2517D120 → 120
    return int(t[-3:]) if t[-4] in "DP" else int(''.join(filter(str.isdigit,t[-4:])))

# ───────── Load one strike ────────────────────────────────────────────
def load_pair(call_tbl, put_tbl, strike):
    with sqlite3.connect(CALL_DB) as con:
        dfC = (pd.read_sql(f"SELECT V_time,C_bid,C_ask,S_last FROM {call_tbl}",
                           con, parse_dates=["V_time"])
               .sort_values("V_time")
               .rename(columns={"C_bid":"bid_C","C_ask":"ask_C"}))
    with sqlite3.connect(PUT_DB) as con:
        dfP = (pd.read_sql(f"SELECT V_time,P_bid,P_ask,S_last FROM {put_tbl}",
                           con, parse_dates=["V_time"])
               .sort_values("V_time")
               .rename(columns={"P_bid":"bid_P","P_ask":"ask_P"}))

    df = (pd.merge(dfC, dfP, on="V_time", suffixes=("","_P"))
            .drop(columns=["S_last_P"]))
    df["trade_date"] = df["V_time"].dt.normalize()
    df["tau"]        = (EXPIRY - df["trade_date"]).dt.days/252
    df = df[df.tau>0].reset_index(drop=True)
    if df.empty: return df

    df["iv_C"] = df.apply(lambda r: iv_brent(r.S_last,strike,r.tau,RISK_FREE,r.bid_C,"call"),axis=1)
    df["iv_P"] = df.apply(lambda r: iv_brent(r.S_last,strike,r.tau,RISK_FREE,r.bid_P,"put"), axis=1)
    d1C = bs_d1(df.S_last,strike,df.tau,RISK_FREE,df.iv_C)
    d1P = bs_d1(df.S_last,strike,df.tau,RISK_FREE,df.iv_P)
    df["delta_port"] = -norm.cdf(d1C) - (norm.cdf(d1P)-1)
    df["delta_jump"] = df["delta_port"].diff().abs()
    df["speed"]      = df.S_last.pct_change()
    return df.fillna(0.0)

def label_rows(df):
    y = np.zeros(len(df),dtype=int)
    last = df.V_time.iat[0]-timedelta(minutes=COOLDOWN_MIN+1)
    for i,r in df.iterrows():
        if (r.V_time-last) < timedelta(minutes=COOLDOWN_MIN): continue
        if abs(r.delta_port)>=EPS_DELTA and r.delta_jump>=EPS_JUMP and \
           (r.speed<=-SPEED_DN or r.speed>=SPEED_UP):
            y[i]=1; last=r.V_time
    return pd.Series(y,index=df.index,name="y")

# ───────── 1. discover common strikes ─────────────────────────────────
call_map={strike_from(t):t for t in tables_in(CALL_DB) if t.startswith("NVDA")}
put_map ={strike_from(t):t for t in tables_in(PUT_DB)  if t.startswith("NVDA")}
common   = sorted(set(call_map)&set(put_map))
if not common: raise RuntimeError("No common strikes!")

print("Common strikes:", common)

# ───────── 2. build dataset ───────────────────────────────────────────
feat = ["delta_port","delta_jump","speed","minutes_since_open","strike","tick_index"]
frames=[]
for strike in common:
    df = load_pair(call_map[strike], put_map[strike], strike)
    if df.empty:
        print(f"  • skip {strike} (all ticks after expiry)"); continue
    df["minutes_since_open"] = (df.V_time-df.V_time.iloc[0]).dt.total_seconds()/60
    df["strike"]     = strike
    df["tick_index"] = np.arange(len(df))
    df["y"] = label_rows(df)
    frames.append(df[feat+["y"]])

if not frames: raise RuntimeError("No rows available for training!")

full = (pd.concat(frames,ignore_index=True)
          .replace([np.inf,-np.inf],np.nan)
          .dropna(subset=feat)
          .reset_index(drop=True))
print("Samples after NaN/inf drop:", len(full))

# ───────── 3. split by strike ─────────────────────────────────────────
strikes = sorted(map(int, full.strike.unique()))
train_strk, val_strk = strikes[:N_TRAIN_STRIKES], strikes[N_TRAIN_STRIKES:]
train = full[full.strike.isin(train_strk)]
val   = full[full.strike.isin(val_strk)]

X_train, y_train = train[feat].values, train["y"].values
X_val,   y_val   = val[feat].values,   val["y"].values

# ───────── 4. fit model ───────────────────────────────────────────────
pipe = Pipeline([("scaler",StandardScaler()),
                 ("clf",   LogisticRegression(max_iter=500,class_weight="balanced"))])

with warnings.catch_warnings():
    warnings.simplefilter("ignore",category=UserWarning)
    pipe.fit(X_train,y_train)

print("\nValidation by-strike report:")
print(classification_report(y_val, pipe.predict(X_val), digits=3))

# ───────── 5. save artefacts ──────────────────────────────────────────
joblib.dump(pipe,"hedge_model.pkl")
meta = dict(features=feat,
            train_strikes=[int(s) for s in train_strk],   # ← cast to int
            val_strikes =[int(s) for s in val_strk])      # ← cast to int
with open("hedge_meta.json","w") as f:
    json.dump(meta, f, indent=2)

print("Saved hedge_model.pkl & hedge_meta.json")
