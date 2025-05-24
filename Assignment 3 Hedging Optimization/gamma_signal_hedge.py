#!/usr/bin/env python3
"""
NVDA Short‑Straddle • Δ‑hedge experiments
========================================
• Short 1 call + 1 put (same strike/expiry)
• Solve Black‑Scholes IV every tick for both legs
• Compare four P&L curves:
    1. Unhedged
    2. Static Δ hedge opened at first tick
    3. Static Δ hedge opened at the best tick in hindsight (scan)
    4. γ‑state hedge (enter when |Γ| ≥ γ_thresh, exit when |Γ| < γ_thresh)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm
from scipy.optimize import brentq

# ────────────────────────────────────────────────────────────────
# Parameters
# ────────────────────────────────────────────────────────────────
strike = 600                              # common strike for call & put

DB_PATH_C = Path("META_call.db")
DB_PATH_P  = Path("META_put.db")
TABLE_C = f"META2516E{strike}"
TABLE_P = f"META2516Q{strike}"
r = 0.06                                   # annual risk‑free rate
expiry_date = pd.Timestamp("2025-05-16")  # contract expiry
EPS_DELTA = 0.05
EPS_GAMMA = 0.005                   # |Γ| trigger for γ‑state hedge

# ────────────────────────────────────────────────────────────────
# Black‑Scholes helpers
# ────────────────────────────────────────────────────────────────

def bs_d1(S, K, tau, r_, sigma):
    """Vectorised Black‑Scholes *d1* (works with scalars or Series/ndarray)"""
    return (np.log(S / K) + (r_ + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))


def bs_price(S, K, tau, r_, sigma, typ="call"):
    d1 = bs_d1(S, K, tau, r_, sigma)
    d2 = d1 - sigma * np.sqrt(tau)
    disc = K * np.exp(-r_ * tau)
    if typ == "call":
        return S * norm.cdf(d1) - disc * norm.cdf(d2)
    else:
        return disc * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol_brent(
    S: float,
    K: float,
    tau: float,
    r_: float,
    mkt: float,
    typ: str = "call",
    lo: float = 1e-6,
    hi: float = 5.0,
    tol: float = 1e-8,
) -> float:
    """Root‑find σ that matches BS price to *mkt*. Returns np.nan on failure."""
    if tau <= 0 or S <= 0 or mkt <= 0:
        return np.nan
    intrinsic = max(S - K * np.exp(-r_ * tau), 0) if typ == "call" else max(K * np.exp(-r_ * tau) - S, 0)
    if mkt < intrinsic:
        return np.nan
    f_lo = bs_price(S, K, tau, r_, lo, typ) - mkt
    f_hi = bs_price(S, K, tau, r_, hi, typ) - mkt
    if f_lo * f_hi > 0:
        return np.nan
    try:
        return brentq(lambda s: bs_price(S, K, tau, r_, s, typ) - mkt, lo, hi, xtol=tol)
    except ValueError:
        return np.nan


def gamma_bs(S, tau, sigma, d1):
    """Vectorised Black‑Scholes Γ (per share). Returns np.nan where invalid."""
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    d1 = np.asarray(d1, dtype=float)

    gamma = np.full_like(S, np.nan)
    mask = (S > 0) & (tau > 0) & (sigma > 0)
    gamma[mask] = norm.pdf(d1[mask]) / (S[mask] * sigma[mask] * np.sqrt(tau[mask]))
    return gamma

# ────────────────────────────────────────────────────────────────
# γ‑state hedge logic
# ────────────────────────────────────────────────────────────────

def run_gamma_aware_delta_hedge(df: pd.DataFrame, eps_delta=0.05, eps_gamma=0.002) -> pd.DataFrame:
    position = 0.0
    tgt, qty, cf = [], [], []

    for i in range(1, len(df)):
        d_now = df.delta_port.iloc[i]
        g_now = df.gamma_port.iloc[i]
        delta_S = df.S_last.iloc[i] - df.S_last.iloc[i - 1]
        delta_change_est = abs(g_now * delta_S)

        if abs(d_now) >= eps_delta and delta_change_est >= eps_gamma:
            target = -d_now * 100
            trade = target - position
            position = target
        else:
            target = position
            trade = 0.0

        tgt.append(target)
        qty.append(trade)
        cf.append(-trade * df.S_last.iloc[i])

    tgt.insert(0, -df.delta_port.iloc[0] * 100)
    qty.insert(0, tgt[0])
    cf.insert(0, -qty[0] * df.S_last.iloc[0])

    df = df.copy()
    df["target_shares"] = tgt
    df["trade_qty"] = qty
    df["cash_flow"] = cf
    df["cum_cash"] = df["cash_flow"].cumsum()
    df["hedge_value"] = df["target_shares"] * df["S_last"]
    df["hedge_pnl_gamma"] = df["cum_cash"] + df["hedge_value"]
    df["pnl_gamma_total"] = df["pnl_port"] + df["hedge_pnl_gamma"]

    return df

# ────────────────────────────────────────────────────────────────
# Load option tables
# ────────────────────────────────────────────────────────────────

with sqlite3.connect(DB_PATH_C) as con:
    df_C = (
        pd.read_sql(f"SELECT V_time, C_bid, C_ask, S_last FROM {TABLE_C}", con, parse_dates=["V_time"])
          .sort_values("V_time")
          .rename(columns={"C_bid": "bid_C", "C_ask": "ask_C"})
    )

with sqlite3.connect(DB_PATH_P) as con:
    df_P = (
        pd.read_sql(f"SELECT V_time, P_bid, P_ask, S_last FROM {TABLE_P}", con, parse_dates=["V_time"])
          .sort_values("V_time")
          .rename(columns={"P_bid": "bid_P", "P_ask": "ask_P"})
    )

# Merge and clean

df = pd.merge(df_C, df_P, on="V_time", suffixes=("", "_P")).drop(columns=["S_last_P"])

df["trade_date"] = df.V_time.dt.normalize()
df["tau"] = (expiry_date - df.trade_date).dt.days / 252

df = df[df.tau > 0].copy()

# ────────────────────────────────────────────────────────────────
# IV solves
# ────────────────────────────────────────────────────────────────

df["iv_C"] = df.apply(lambda row: implied_vol_brent(row.S_last, strike, row.tau, r, row.ask_C, "call"), axis=1)
df["iv_P"] = df.apply(lambda row: implied_vol_brent(row.S_last, strike, row.tau, r, row.ask_P, "put"), axis=1)

# ────────────────────────────────────────────────────────────────
# Greeks
# ────────────────────────────────────────────────────────────────

df["d1_C"] = bs_d1(df.S_last, strike, df.tau, r, df.iv_C)
df["d1_P"] = bs_d1(df.S_last, strike, df.tau, r, df.iv_P)

delta_long_call = norm.cdf(df.d1_C)
delta_long_put  = norm.cdf(df.d1_P) - 1

df["delta_port"] = -(delta_long_call + delta_long_put)

gamma_long_call = gamma_bs(df.S_last, df.tau, df.iv_C, df.d1_C)
gamma_long_put  = gamma_bs(df.S_last, df.tau, df.iv_P, df.d1_P)

df["gamma_port"] = -(gamma_long_call + gamma_long_put)

# ────────────────────────────────────────────────────────────────
# P&L calculations
# ────────────────────────────────────────────────────────────────

open_bid_C = df.bid_C.iloc[0]
open_bid_P = df.bid_P.iloc[0]

df["pnl_call"] = open_bid_C - df.ask_C
df["pnl_put"]  = open_bid_P - df.ask_P
df["pnl_port"] = (df.pnl_call + df.pnl_put) * 100

# Static Δ hedge @ first tick
open_stock  = df.S_last.iloc[0]
delta0      = df.delta_port.iloc[0]

df["hedge_pnl_first"]  = -(df.S_last - open_stock) * (delta0 * 100)
df["pnl_hedged_first"] = df.pnl_port + df.hedge_pnl_first

# Static Δ hedge scan (best start)
S_end = df.S_last.iloc[-1]
df["hedge_pnl_scan"]  = -(S_end - df.S_last) * (df.delta_port * 100)
df["pnl_scan_total"] = df.pnl_port + df.hedge_pnl_scan
best_idx = df.pnl_scan_total.idxmax()
best_row = df.loc[best_idx]
# γ-state hedge
# ────────────────────────────────────────────────────────────────
df = run_gamma_aware_delta_hedge(df, eps_delta=EPS_DELTA, eps_gamma=EPS_GAMMA)

# -------------- insert these two lines just below the call --------------
pct_hedged = (df.target_shares != 0).mean() * 100        # % of ticks hedged
# ------------------------------------------------------------------------
# ────────────────────────────────────────────────────────────────
# Console summary
# ────────────────────────────────────────────────────────────────

print("===== NVDA short‑straddle hedge summary ===================")
print(f"Strike (K)                    : {strike}")
print(f"First‑tick portfolio Δ        : {delta0: .4f}")
print("------------------------------------------------------------")
print(f"Optimum static‑hedge start    : {best_row.V_time}")
print(f"  Final P&L if hedged there   : {best_row.pnl_scan_total: .2f} $")
print("------------------------------------------------------------")
print(f"Final unhedged P&L            : {df.pnl_port.iloc[-1]: .2f} $")
print(f"Final first‑tick‑hedge P&L    : {df.pnl_hedged_first.iloc[-1]: .2f} $")
print(f"Final γ‑state‑hedge P&L       : {df.pnl_gamma_total.iloc[-1]: .2f} $")
print("============================================================")
print(f"Pct. of ticks hedged (γ-state): {pct_hedged: .2f}%")
print("============================================================")
# ────────────────────────────────────────────────────────────────
# Plot
# ────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(df.V_time, df.pnl_port,          label="Unhedged P&L",               lw=0.9)
ax.plot(df.V_time, df.pnl_hedged_first,  label="Static hedge @ first tick", lw=0.9)
ax.plot(df.V_time, df.pnl_scan_total,    label="Static hedge if starts *here*", lw=0.9, alpha=0.6)
ax.plot(df.V_time, df.pnl_gamma_total,   label=f"γ‑state hedge", lw=1.1, color="purple")

ax.axvline(best_row.V_time, color="black", ls="--", lw=0.8, label="Best static‑hedge start")

ax2 = ax.twinx()
ax2.plot(df.V_time, df.S_last, color="grey", alpha=0.3, label="Stock price")
ax2.set_ylabel("Stock price ($)")

major = mdates.AutoDateLocator(minticks=6, maxticks=10)
ax.xaxis.set_major_locator(major)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(major))
ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune="both"))

ax.grid(which="major", axis="y", ls="--", lw=0.6, alpha=0.4)
ax.grid(which="major", axis="x", ls=":", lw=0.6, alpha=0.3)

ax.set_title(f"NVDA {strike} Short‑straddle — Δ‑hedge experiments")
ax.set_ylabel("P&L Total ($)")
ax.legend(loc="upper left")
ax2.legend(loc="upper right", frameon=False)

fig.tight_layout()
fig.savefig("straddle_hedge_comparison.png", dpi=300)
plt.show()

