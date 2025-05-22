#!/usr/bin/env python
# =======================================================================
#   NVDA Call – Delta-hedge scanner with *per-tick* implied volatility
#   ---------------------------------------------------------------
#   • Re-prices IV at every tick from the option ask (change if desired).
#   • Computes d1, Delta, Gamma from that IV and time-to-expiry (τ).
#   • Shows three P&L curves:
#         – Unhedged                     (buy call, do nothing)
#         – Hedged at the *first* tick   (classic static hedge)
#         – “If I started hedging here”  (scan curve)
#   • Reports the timestamp that maximises the scan P&L.
# =======================================================================

import sqlite3, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates   as mdates
from matplotlib.ticker import MaxNLocator
from pathlib   import Path
from scipy.stats   import norm
from scipy.optimize import brentq

# -------------------- Parameters ---------------------------------------
strike       = 120
DB_PATH      = Path("NVDA_call.db")
TABLE        = f"NVDA2517D{strike}"
option_type  = "call"
r            = 0.06                         # annual risk-free rate
expiry_date  = pd.Timestamp("2025-04-17")   # option expiry (adjust if needed)

# -------------------- Black–Scholes helpers ----------------------------
def black_scholes_d1(S, K, tau, r, sigma):
    """d1 with year-fraction tau."""
    return (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))

def black_scholes_price(S, K, tau, r, sigma, option_type="call"):
    d1  = black_scholes_d1(S, K, tau, r, sigma)
    d2  = d1 - sigma * np.sqrt(tau)
    disc = K * np.exp(-r * tau)
    if option_type == "call":
        return S * norm.cdf(d1) - disc * norm.cdf(d2)
    else:  # put
        return disc * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_vol_brent(S, K, tau, r, mkt, typ="call",
                      lo=1e-6, hi=5.0, tol=1e-8):
    if tau <= 0 or mkt <= 0 or S <= 0:          # basic sanity
        return np.nan
    intr = max(S - K*np.exp(-r*tau), 0) if typ=="call" else \
           max(K*np.exp(-r*tau) - S,    0)
    if mkt < intr:                              # violated lower bound
        return np.nan
    f_lo = black_scholes_price(S, K, tau, r, lo, typ) - mkt
    f_hi = black_scholes_price(S, K, tau, r, hi, typ) - mkt
    if f_lo * f_hi > 0:                         # no sign change
        return np.nan
    try:
        return brentq(lambda s: black_scholes_price(S,K,tau,r,s,typ)-mkt,
                      lo, hi, xtol=tol)
    except ValueError:
        return np.nan


def gamma_bs(S, tau, sigma, d1):
    return norm.pdf(d1) / (S * sigma * np.sqrt(tau))

# -------------------- Load table ---------------------------------------
with sqlite3.connect(DB_PATH) as con:
    df = (
        pd.read_sql(f"SELECT V_time, C_bid, C_ask, S_last FROM {TABLE}",
                    con, parse_dates=["V_time"])
          .sort_values("V_time")
    )

df["trade_date"] = df["V_time"].dt.normalize()

# -------------------- First-tick benchmarks ----------------------------
open_ask   = df["C_ask"].iloc[0]
open_stock = df["S_last"].iloc[0]

first_trade_date = df["trade_date"].iloc[0]
if expiry_date <= first_trade_date:
    raise ValueError("Expiry date must be after first trade date")

tau0  = (expiry_date - first_trade_date).days / 252          # first-tick τ
iv0   = implied_vol_brent(open_stock, strike, tau0, r, open_ask, option_type)
d10   = black_scholes_d1(open_stock, strike, tau0, r, iv0)
delta0 = norm.cdf(d10)
gamma0 = gamma_bs(open_stock, tau0, iv0, d10)

print("===== First-tick stats ====================================")
print(f"Strike (K)            : {strike}")
print(f"Business days to exp  : {(expiry_date-first_trade_date).days}")
print(f"τ₀ (years)            : {tau0: .6f}")
print(f"IV₀                   : {iv0: .4f}")
print(f"d1₀                   : {d10: .4f}")
print(f"Δ₀                    : {delta0: .4f}")
print(f"γ₀                    : {gamma0: .6f}")
print("===========================================================\n")

# -------------------- Baseline P&L series (hedge at first tick) ---------
df["open_ask"]    = open_ask
df["pnl_tick"]    = df["C_bid"] - df["open_ask"]
df["hedge_pnl1"]  = -(df["S_last"] - open_stock) * delta0        # hedge @ first tick
df["pnl_hedged1"] = df["pnl_tick"] + df["hedge_pnl1"]

# -------------------- Per-tick τ (year-frac to expiry) ------------------
df["tau"] = (expiry_date - df["trade_date"]).dt.days / 252
df = df[df["tau"] > 0].copy()       # keep only pre-expiry rows

# -------------------- Per-tick implied volatility -----------------------
def row_iv(row):
    return implied_vol_brent(row.S_last, strike, row["tau"], r, row.C_ask, option_type)

df["iv"] = df.apply(row_iv, axis=1)

# -------------------- d1, Δ, Γ for every tick ---------------------------
df["d1"]    = black_scholes_d1(df["S_last"], strike, df["tau"], r, df["iv"])
df["delta"] = norm.cdf(df["d1"])
df["gamma"] = gamma_bs(df["S_last"], df["tau"], df["iv"], df["d1"])

# -------------------- Scan hedge-PnL curve ------------------------------
S_end = df["S_last"].iloc[-1]               # flatten hedge at final stock price
df["hedge_pnl_scan"] = -(S_end - df["S_last"]) * df["delta"]
df["pnl_scan_total"] = df["pnl_tick"] + df["hedge_pnl_scan"]

best_idx = df["pnl_scan_total"].idxmax()
best     = df.loc[best_idx]

print("===== Optimal hedge-start tick ============================")
print(f"Timestamp              : {best.V_time}")
print(f"τ (years)              : {best.tau: .6f}")
print(f"IV                     : {best.iv: .4f}")
print(f"d1                     : {best.d1: .4f}")
print(f"Δ                      : {best.delta: .4f}")
print(f"Max total PnL ($)      : {best.pnl_scan_total: .4f}")
print("===========================================================\n")

# -------------------- Correlations --------------------------------------
corr_unhedged = df["pnl_tick"      ].corr(df["S_last"])
corr_hedged1  = df["pnl_hedged1"   ].corr(df["S_last"])
print(f"Corr (unhedged PnL , S)         : {corr_unhedged: .4f}")
print(f"Corr (hedged @ first , S)       : {corr_hedged1 : .4f}")

# -------------------- Plot ------------------------------------------------
fig, ax = plt.subplots(figsize=(15, 6))

ax.plot(df["V_time"], df["pnl_tick"],        label="Unhedged PnL", lw=0.9)
ax.plot(df["V_time"], df["pnl_hedged1"],     label="PnL hedged @ first tick", lw=0.9)
ax.plot(df["V_time"], df["pnl_scan_total"],  label="PnL if hedge starts *here*", lw=0.9, alpha=0.6)
ax.axvline(best.V_time, color="black", ls="--", lw=0.8, label="Optimal hedge-start")

ax2 = ax.twinx()
ax2.plot(df["V_time"], df["S_last"], label="Stock price", color="grey", alpha=0.3)
ax2.set_ylabel("Stock price ($)")

major = mdates.AutoDateLocator(minticks=6, maxticks=10)
ax.xaxis.set_major_locator(major)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(major))
ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune="both"))

ax.grid(which="major", axis="y", ls="--", lw=0.6, alpha=0.4)
ax.grid(which="major", axis="x", ls=":",  lw=0.6, alpha=0.3)

ax.set_title("NVDA Call P&L — delta-hedge scan "
             f"(γ₀={gamma0:.3f}, Δ₀={delta0:.3f})")
ax.set_ylabel("PnL per contract ($)")
ax.legend(loc="upper left")
ax2.legend(loc="upper right", frameon=False)

fig.tight_layout()
fig.savefig("call_hedge_scan.png", dpi=300)
plt.show()
