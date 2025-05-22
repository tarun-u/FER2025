#!/usr/bin/env python3
# ================================================================
#   Gamma–Driven Delta-Hedge Simulator
#   – Long 1 NVDA call, strike = 100
#   – Hedge to −Δ shares only while Γ ≥ 0.004
#   – Flatten hedge when Γ < 0.004
# ================================================================

import sqlite3, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates   as mdates
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from scipy.stats  import norm
from scipy.optimize import brentq

# -------------------- Parameters --------------------------------
strike        = 100
DB_PATH       = Path("NVDA_call.db")
TABLE         = f"NVDA2517D{strike}"       # e.g. NVDA2517D100
option_type   = "call"
r             = 0.06                       # annual risk-free rate
expiry_date   = pd.Timestamp("2025-04-17")
contract_size = 1                          # 1 → $ per option
gamma_thresh  = 0.004                      # Γ threshold

# -------------------- Black–Scholes helpers ---------------------
def black_scholes_d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    d1 = black_scholes_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_vol_brent(S, K, T, r, mkt_price,
                      option_type="call",
                      sigma_low=1e-6, sigma_high=5.0, tol=1e-8):
    f = lambda sigma: black_scholes_price(S, K, T, r, sigma, option_type) - mkt_price
    try:
        return brentq(f, sigma_low, sigma_high, xtol=tol)
    except ValueError:
        return np.nan

def calculate_gamma(S, T, sigma, d1):
    gamma = (np.exp(-d1**2/2)/((2 * np.pi)**0.5))/(S * sigma * (T ** 0.5))
    return gamma

# -------------------- Core simulator ----------------------------
def simulate_gamma_threshold_hedge(df,
                                   strike,
                                   r,
                                   expiry_date,
                                   gamma_thresh,
                                   option_type="call",
                                   contract_size=1):

    # ── first-tick constants ─────────────────────────────────────
    open_ask   = df["C_ask"].iloc[0] * contract_size
    open_stock = df["S_last"].iloc[0]

    T0 = (expiry_date - df["trade_date"].iloc[0]).days / 252
    iv = implied_vol_brent(open_stock, strike, T0, r,
                           open_ask/contract_size, option_type)
    if np.isnan(iv):
        raise ValueError("IV solver failed at first tick")

    # ── Greeks for every tick (σ frozen at iv) ───────────────────
    df = df.copy()
    df["T"]     = (expiry_date - df["trade_date"]).dt.days / 252
    df["d1"]    = black_scholes_d1(df["S_last"], strike, df["T"], r, iv)
    df["delta"] = norm.cdf(df["d1"])
    df["gamma"] = calculate_gamma(df["S_last"], df["T"], iv, df["d1"])

    # ── Hedge rule ───────────────────────────────────────────────
    # target shares: −Δ when Γ ≥ threshold, else 0
    df["target_shares"] = np.where(df["gamma"] >= gamma_thresh,
                                   -df["delta"], 0.0)

    # ── Trading ledger ───────────────────────────────────────────
    df["trade_qty"] = df["target_shares"].diff().fillna(df["target_shares"])
    df["cash_flow"] = -df["trade_qty"] * df["S_last"]
    df["cum_cash"]  = df["cash_flow"].cumsum()

    df["hedge_value"] = df["target_shares"] * df["S_last"]
    df["hedge_pnl"]   = df["cum_cash"] + df["hedge_value"]

    # ── Option & total P&L ──────────────────────────────────────
    df["option_pnl"] = (df["C_bid"] * contract_size) - open_ask
    df["total_pnl"]  = df["option_pnl"] + df["hedge_pnl"]

    # ── Final flatten (safety) ──────────────────────────────────
    final_price = df["S_last"].iloc[-1]
    final_qty   = -df["target_shares"].iloc[-1]
    final_cf    = -final_qty * final_price
    df.loc[df.index[-1], "trade_qty"] += final_qty
    df.loc[df.index[-1], "cash_flow"] += final_cf

    df["cum_cash"]    = df["cash_flow"].cumsum()
    df["hedge_value"] = df["target_shares"] * df["S_last"]
    df["hedge_pnl"]   = df["cum_cash"] + df["hedge_value"]
    df["total_pnl"]   = df["option_pnl"] + df["hedge_pnl"]

    # first-tick Greek snapshot
    gamma0 = df["gamma"].iloc[0]
    delta0 = df["delta"].iloc[0]
    return df, iv, delta0, gamma0

# -------------------- Load table --------------------------------
with sqlite3.connect(DB_PATH) as con:
    df_raw = (
        pd.read_sql(f"SELECT V_time, C_bid, C_ask, S_last FROM {TABLE}",
                    con, parse_dates=["V_time"])
          .sort_values("V_time")
    )
df_raw["trade_date"] = df_raw["V_time"].dt.normalize()

# -------------------- Run simulation ----------------------------
(df,
 iv,
 delta0,
 gamma0) = simulate_gamma_threshold_hedge(df_raw,
                                          strike=strike,
                                          r=r,
                                          expiry_date=expiry_date,
                                          gamma_thresh=gamma_thresh,
                                          option_type=option_type,
                                          contract_size=contract_size)

print("\n===== Summary ============================================")
print(f"Strike                   : {strike}")
print(f"Γ threshold              : {gamma_thresh}")
print(f"First-tick IV            : {iv: .4f}")
print(f"First-tick Δ             : {delta0: .4f}")
print(f"First-tick Γ             : {gamma0: .6f}")
print(f"Final P&L (Γ-rule hedge) : {df['total_pnl'].iloc[-1]: .2f} $")
print("==========================================================\n")

# -------------------- Hedge utilisation stats -------------------
hedged_time = (df["target_shares"] != 0).mean()
print(f"Fraction of ticks hedged : {hedged_time: .2%}")

# -------------------- Plot --------------------------------------
fig, ax = plt.subplots(figsize=(15, 6))

ax.plot(df["V_time"], df["option_pnl"], label="Option P&L (no hedge)", lw=0.9)
ax.plot(df["V_time"], df["total_pnl"],  label="Γ-rule Δ-hedged P&L",    lw=1.1)

ax.set_ylabel("P&L per contract ($)")

ax2 = ax.twinx()
ax2.plot(df["V_time"], df["S_last"], color="grey", alpha=0.3,
         label="Stock price")
ax2.set_ylabel("Stock price")

major = mdates.AutoDateLocator(minticks=6, maxticks=10)
ax.xaxis.set_major_locator(major)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(major))
ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune="both"))

ax.grid(which="major", axis="y", ls="--", lw=0.6, alpha=0.4)
ax.grid(which="major", axis="x", ls=":",  lw=0.6, alpha=0.3)

ax.set_title(f"NVDA {strike}C — Long call, Δ hedge only while Γ ≥ {gamma_thresh}"
             f"\n(IV₀={iv:.3f}, Δ₀={delta0:.3f}, Γ₀={gamma0:.6f})")
ax.legend(loc="upper left")
ax2.legend(loc="upper right", frameon=False)

fig.tight_layout()
fig.savefig("call_gamma_rule_hedge.png", dpi=300)
plt.show()
