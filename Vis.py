import sqlite3, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DB_PATH = Path("NVDA_call.db")
TABLE = "NVDA2517D100"  # or whatever contract you want

# Load data
with sqlite3.connect(DB_PATH) as con:
    df = pd.read_sql(f"""
        SELECT V_time, C_bid, C_ask, C_last
        FROM [{TABLE}]
        ORDER BY V_time ASC
    """, con, parse_dates=["V_time"])

# Basic sanity check
if df.empty:
    print("No data found.")
    exit()

# Plotting
plt.figure(figsize=(14, 6))
plt.plot(df["V_time"], df["C_bid"], label="Bid", linestyle='--', color='green')
plt.plot(df["V_time"], df["C_ask"], label="Ask", linestyle='--', color='red')
plt.plot(df["V_time"], df["C_last"], label="Last Trade", color='blue')

plt.xlabel("Time")
plt.ylabel("Option Price")
plt.title(f"Bid/Ask/Last Over Time for {TABLE}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


png_path = f"{TABLE}_price_plot.png"
plt.savefig(png_path)
print(f"✓ Plot saved as {png_path}")