import sqlite3, pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm
from scipy.optimize import brentq

contract = 120
DB_PATH = Path("NVDA_put.db")
TABLE   = "NVDA2517P" + str(contract)

# ─── Black-Scholes and IV Functions ─────────────────────────────────────

def black_scholes_d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    d1 = black_scholes_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def calculate_vega(S, K, T, r, sigma):
    d1 = black_scholes_d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)

def implied_vol_brent(S,K,T,r,market_price,option_type="call",
                      sigma_low=1e-6, sigma_high=5.0, tol=1e-8):
    f = lambda sigma: black_scholes_price(S,K,T,r,sigma,option_type) - market_price
    try:
        return brentq(f, sigma_low, sigma_high, xtol=tol)
    except ValueError:
        return np.nan

# ─── Load Data ──────────────────────────────────────────────────────────

con = sqlite3.connect(DB_PATH)
df = pd.read_sql(
    f"SELECT V_time, P_bid, P_ask, S_last FROM {TABLE}",
    con,
    parse_dates=["V_time"]
).sort_values("V_time")
con.close()

df["trade_date"] = df["V_time"].dt.normalize()

# ─── Get First Tick Values ──────────────────────────────────────────────

open_ask = df["P_ask"].iloc[0]
open_stock = df["S_last"].iloc[0]

# Constants
K = contract
r = 0.06
first_trade_date = df["trade_date"].iloc[0]
expiry_date       = pd.Timestamp("2025-04-15")

if expiry_date <= first_trade_date:
    raise ValueError("Expiry date must be after first trade date")

days_to_expiry = (expiry_date - first_trade_date).days
T = days_to_expiry / 252
option_type = "put"

# ─── Calculate IV and Delta Only Once at First Tick ─────────────────────

iv = implied_vol_brent(S=open_stock, K=K, T=T, r=r, market_price=open_ask, option_type=option_type)
d1 = black_scholes_d1(S=open_stock, K=K, T=T, r=r, sigma=iv)
delta = norm.cdf(d1) - 1  # ← Put delta

print(f"Strike K parsed from table name: {K}")
print(f"Business days to expiry: {days_to_expiry}")
print(f"Implied Volatility: {iv:.4f}")
print(f"d1: {d1:.4f}")
print(f"Put Delta (First Tick): {delta:.4f}")

# ─── PnL Calculations ───────────────────────────────────────────────────

df["open_ask"] = open_ask
df["pnl_tick"] = df["P_bid"] - df["open_ask"]

# Apply hedge ONLY based on initial delta and stock price
df["hedge_pnl"] = -(df["S_last"] - open_stock) * delta
df["pnl_hedged"] = df["pnl_tick"] + df["hedge_pnl"]

# ─── Correlation Checks ─────────────────────────────────────────────────

corr = df["pnl_tick"].corr(df["S_last"])
corr_hedged = df["pnl_hedged"].corr(df["S_last"])

print(f"Correlation (Unhedged PnL vs S): {corr:.4f}")
print(f"Correlation (Hedged PnL vs S):   {corr_hedged:.4f}")
print(df)

# ─── Plot ───────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(15, 6))

ax.plot(df["V_time"], df["pnl_tick"], label="Unhedged PnL", lw=0.8)
ax.plot(df["V_time"], df["pnl_hedged"], label="Delta-Hedged PnL (first tick only)", lw=0.8)
ax.plot(df["V_time"], df["S_last"], label="Stock Price", alpha=0.3)

major_loc = mdates.AutoDateLocator(minticks=6, maxticks=10)
ax.xaxis.set_major_locator(major_loc)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(major_loc))
ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune="both"))

ax.grid(False)
ax.grid(which="major", axis="y", ls="--", lw=0.6, alpha=0.4)
ax.grid(which="major", axis="x", ls=":",  lw=0.6, alpha=0.3)

ax.set_title("Intraday Put Option PnL with Delta Hedge (1st Tick Only)")
ax.set_ylabel("PnL per contract ($)")
ax.legend()
fig.tight_layout()

fig.savefig("put_hedge.png", dpi=300)
plt.show()
