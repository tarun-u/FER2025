import sqlite3, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

CALL_DB_PATH = Path("NVDA_call.db")
PUT_DB_PATH  = Path("NVDA_put.db")
CALL_TABLE = "NVDA2517D100"  # Update if your table name differs
PUT_TABLE  = "NVDA2517P100"
STRIKE     = 100  # optional, if you want to print/use it later

def compute_straddle_pnl(call_df, put_df):
    call_df = call_df.sort_values("V_time").copy()
    put_df  = put_df.sort_values("V_time").copy()

    call_df["trade_date"] = call_df["V_time"].dt.normalize()
    put_df["trade_date"]  = put_df["V_time"].dt.normalize()

    # Get daily open premiums (first ask)
    call_open = call_df.groupby("trade_date")["C_ask"].first().rename("call_ask_open")
    put_open  = put_df.groupby("trade_date")["P_ask"].first().rename("put_ask_open")

    # Get daily close prices (last bid)
    call_close = call_df.groupby("trade_date")["C_bid"].last().rename("call_bid_close")
    put_close  = put_df.groupby("trade_date")["P_bid"].last().rename("put_bid_close")

    # Merge
    daily = pd.concat([call_open, put_open, call_close, put_close], axis=1).dropna()

    # Calculate straddle PnL
    daily["premium_paid"] = daily["call_ask_open"] + daily["put_ask_open"]
    daily["total_value"]  = daily["call_bid_close"] + daily["put_bid_close"]
    daily["pnl_straddle"] = (daily["total_value"] - daily["premium_paid"]).round(2)

    return daily.reset_index()

# ─── Load call and put data from separate DBs ────────────────────────────────
with sqlite3.connect(CALL_DB_PATH) as call_con, sqlite3.connect(PUT_DB_PATH) as put_con:
    call_df = pd.read_sql(f"SELECT V_time, C_bid, C_ask FROM [{CALL_TABLE}]", call_con, parse_dates=["V_time"])
    put_df  = pd.read_sql(f"SELECT V_time, P_bid, P_ask FROM [{PUT_TABLE}]",  put_con,  parse_dates=["V_time"])

straddle_pnl = compute_straddle_pnl(call_df, put_df)
straddle_pnl.to_csv("straddle_pnl.csv", index=False)
print("✓ Straddle P&L saved to straddle_pnl.csv")

# ─── Plot PnL over time ──────────────────────────────────────────────────────
plt.figure(figsize=(12, 6))
plt.plot(straddle_pnl["trade_date"], straddle_pnl["pnl_straddle"], marker='o', label="Straddle P&L")
plt.axhline(0, color='gray', linestyle='--')

plt.title("Daily Straddle P&L Over Time")
plt.xlabel("Trade Date")
plt.ylabel("P&L ($)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save plot
plt.savefig("straddle_pnl_plot.png", dpi=300)
plt.show()
print("✓ Plot saved to straddle_pnl_plot.png")
