#!/usr/bin/env python3
import sqlite3, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from scipy.stats import norm
from scipy.optimize import brentq

# -------------------- Parameters ---------------------------------------
strike      = 140
DB_PATH_C   = Path("NVDA_call.db")
DB_PATH_P   = Path("NVDA_put.db")
TABLE_C     = f"NVDA2517D{strike}"
TABLE_P     = f"NVDA2517P{strike}"
rf          = 0.06                                    # annual risk-free rate
expiry_date = pd.Timestamp("2025-04-17")

# -------------------- Black-Scholes helpers ----------------------------
def black_scholes_d1(S, K, tau, r, sigma):
    return (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))

def black_scholes_price(S,K,tau,r,sigma,opt="call"):
    d1 = black_scholes_d1(S,K,tau,r,sigma)
    d2 = d1 - sigma*np.sqrt(tau)
    disc = K*np.exp(-r*tau)
    if opt=="call":
        return S*norm.cdf(d1) - disc*norm.cdf(d2)
    return disc*norm.cdf(-d2) - S*norm.cdf(-d1)

def implied_vol_brent(S,K,T,r,market_price,opt="call",
                      sigma_low=1e-6, sigma_high=5.0, tol=1e-8):
    f = lambda sigma: black_scholes_price(S,K,T,r,sigma,opt) - market_price
    try:
        return brentq(f, sigma_low, sigma_high, xtol=tol)
    except ValueError:
        return np.nan

# -------------------- Load tables --------------------------------------
with sqlite3.connect(DB_PATH_C) as con:
    df_C = (pd.read_sql(f"SELECT V_time,C_bid,C_ask,S_last FROM {TABLE_C}",
                        con, parse_dates=["V_time"])
              .sort_values("V_time")
              .rename(columns={"C_bid":"bid_C","C_ask":"ask_C"}))

with sqlite3.connect(DB_PATH_P) as con:
    df_P = (pd.read_sql(f"SELECT V_time,P_bid,P_ask,S_last FROM {TABLE_P}",
                        con, parse_dates=["V_time"])
              .sort_values("V_time")
              .rename(columns={"P_bid":"bid_P","P_ask":"ask_P"}))

df = (pd.merge(df_C, df_P, on="V_time", suffixes=("","_P"))
        .drop(columns=["S_last_P"]))

df["trade_date"] = df["V_time"].dt.normalize()

# -------------------- First tick ---------------------------------------
open_bid_C  = df["bid_C"].iloc[0]
open_bid_P  = df["bid_P"].iloc[0]
open_stock  = df["S_last"].iloc[0]
first_date  = df["trade_date"].iloc[0]
if expiry_date <= first_date:
    raise ValueError("Expiry must be after first trade date")

# -------------------- τ & IVs ------------------------------------------
df["tau"] = (expiry_date - df["trade_date"]).dt.days / 252
df = df[df["tau"] > 0].copy()

df["iv_C"] = df.apply(
    lambda row: implied_vol_brent(row.S_last, strike, row.tau, rf,
                                  row.ask_C, opt="call"), axis=1)
df["iv_P"] = df.apply(
    lambda row: implied_vol_brent(row.S_last, strike, row.tau, rf,
                                  row.ask_P, opt="put"),  axis=1)

# -------------------- Δ -------------------------------------------------
df["d1_C"] = black_scholes_d1(df["S_last"], strike, df["tau"], rf, df["iv_C"])
df["d1_P"] = black_scholes_d1(df["S_last"], strike, df["tau"], rf, df["iv_P"])

delta_long_call = norm.cdf(df["d1_C"])       #  +ve
delta_long_put  = norm.cdf(df["d1_P"]) - 1   #  –ve

df["delta_call_short"] = -delta_long_call    # short call
df["delta_put_short"]  = -delta_long_put     # short put
df["delta_port"]       = df["delta_call_short"] + df["delta_put_short"]

# -------------------- P&L ----------------------------------------------
df["pnl_call"] = open_bid_C - df["bid_C"]
df["pnl_put"]  = open_bid_P - df["bid_P"]
df["pnl_port"] = df["pnl_call"] + df["pnl_put"]

# Static hedge @ first tick
delta0 = df["delta_port"].iloc[0]
df["hedge_pnl1"] = -(df["S_last"] - open_stock) * delta0
df["pnl_hedged1"]= df["pnl_port"] + df["hedge_pnl1"]

# “Hedge starts here” scan
S_end = df["S_last"].iloc[-1]
df["hedge_pnl_scan"] = -(S_end - df["S_last"]) * df["delta_port"]
df["pnl_scan_total"] = df["pnl_port"] + df["hedge_pnl_scan"]
best_row = df.loc[df["pnl_scan_total"].idxmax()]

# -------------------- Summary ------------------------------------------
print("\n===== Short straddle Δ-hedge scan =====")
print(f"Strike                 : {strike}")
print(f"Initial τ (yrs)        : {(expiry_date-first_date).days/252: .6f}")
print(f"Initial portfolio Δ    : {delta0: .4f}")
print("----------------------------------------")
print("Optimal hedge start at :")
print(f"  {best_row.V_time}")
print(f"  Δ at that tick       : {best_row.delta_port: .4f}")
print(f"  P&L if hedged here   : {best_row.pnl_scan_total * 100: .2f} $")
print("========================================\n")

print(f"Corr(unhedged, S)      : {df['pnl_port'].corr(df['S_last']): .4f}")
print(f"Corr(hedged@first, S)  : {df['pnl_hedged1'].corr(df['S_last']): .4f}")

# -------------------- Plot ---------------------------------------------
fig, ax = plt.subplots(figsize=(15,6))
ax.plot(df["V_time"], df["pnl_port"],      label="Naked P&L", lw=0.9)
ax.plot(df["V_time"], df["pnl_hedged1"],   label="Hedged @ first tick", lw=0.9)
ax.plot(df["V_time"], df["pnl_scan_total"],label="If hedge starts here", lw=0.9,
        alpha=0.65)
ax.axvline(best_row.V_time, color="black", ls="--", lw=0.8,
           label="Optimal hedge start")

ax2 = ax.twinx()
ax2.plot(df["V_time"], df["S_last"], color="grey", alpha=0.3, label="Stock")
ax2.set_ylabel("Stock price ($)")

major = mdates.AutoDateLocator(minticks=6, maxticks=10)
ax.xaxis.set_major_locator(major)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(major))
ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune="both"))

ax.grid(axis="y", ls="--", alpha=0.4)
ax.grid(axis="x", ls=":",  alpha=0.3)

ax.set_title(f"NVDA {strike} Short-Straddle — Δ-hedge scan")
ax.set_ylabel("P&L per straddle ($)")
ax.legend(loc="upper left")
ax2.legend(loc="upper right", frameon=False)

fig.tight_layout()
fig.savefig("straddle_delta_hedge_scan.png", dpi=300)
plt.show()
