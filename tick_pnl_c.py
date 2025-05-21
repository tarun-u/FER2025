import sqlite3, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

DB_PATH = Path("NVDA_call.db")
TABLE   = "NVDA2517D100"

con = sqlite3.connect(DB_PATH)

df = pd.read_sql(
    f"SELECT V_time, C_bid, C_ask, S_last FROM {TABLE}",
    con,
    parse_dates=["V_time"]
).sort_values("V_time")

con.close()

df["trade_date"] = df["V_time"].dt.normalize()

open_ask_value = df["C_ask"].iloc[0]

df["open_ask"] = open_ask_value

df["pnl_tick"] = df["C_bid"] - df["open_ask"]

corr = df["pnl_tick"].corr(df["S_last"])
print(f"Correlation (pnl_tick vs S_last): {corr:.4f}")

print(df.head())

fig, ax = plt.subplots(figsize=(15, 6))

ax.plot(df["V_time"], df["pnl_tick"], lw=0.8)
ax.plot(df["V_time"], df["S_last"])

major_loc = mdates.AutoDateLocator(minticks=6, maxticks=10)
ax.xaxis.set_major_locator(major_loc)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(major_loc))

ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune="both"))

ax.grid(False)
ax.grid(which="major", axis="y", ls="--", lw=0.6, alpha=0.4)
ax.grid(which="major", axis="x", ls=":",  lw=0.6, alpha=0.3)

# ─── cosmetics ────────────────────────────────────────────────────────────
ax.set_title("Intraday PnL tick (C_bid - opening ask)")
ax.set_ylabel("PnL per contract ($)")
fig.tight_layout()

fig.savefig("tick_call_pnl", dpi=300)
plt.show()