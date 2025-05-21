#!/usr/bin/env python
from pathlib import Path
import sqlite3, pandas as pd
import numpy as np

DB_PATH = Path("NVDA_call.db")

def compute_pnl(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("V_time").copy()
    df["trade_date"] = df["V_time"].dt.normalize()
    df["open_ask"]   = df.groupby("trade_date")["C_ask"].transform("first")
    df["pnl_tick"]   = df["C_bid"] - df["open_ask"]

    daily = (
        df.groupby("trade_date")
          .agg(open_ask  = ("C_ask", "first"),
               close_bid = ("C_bid", "last"))
          .assign(pnl_eod = lambda x: np.floor((x.close_bid - x.open_ask) * 100) / 100)
          .loc[:, ["pnl_eod"]]
          .reset_index()
    )
    return daily

with sqlite3.connect(DB_PATH, timeout=60) as con:
    con.execute("PRAGMA journal_mode=WAL;")

    tbls = pd.read_sql("""SELECT name
                            FROM sqlite_master
                           WHERE type='table'
                             AND name NOT LIKE 'sqlite_%';""",
                        con)["name"].tolist()

    daily_frames = []
    for tbl in tbls:
        try:
            raw = pd.read_sql(f"SELECT V_time,C_bid,C_ask FROM [{tbl}]",
                              con, parse_dates=["V_time"])
            if raw.empty:
                print(f"[skip] {tbl}: empty.")
                continue

            daily = compute_pnl(raw)
            daily["contract"] = tbl
            daily_frames.append(daily)

            print(f"[done] {tbl}: {len(daily)} sessions.")
        except Exception as e:
            print(f"[error] {tbl}: {e}")

    all_daily = (pd.concat(daily_frames, ignore_index=True, sort=False)
                   .sort_values(["contract", "trade_date"]))

    con.commit()

    all_daily.to_sql("daily_pnl_all", con,
                     if_exists="replace", index=False, method="multi")

all_daily.to_csv("daily_pnl_calls.csv", index=False)
print("P&L written to daily_pnl_all_c table and daily_pnl_all.csv")
