

from pathlib import Path
import sqlite3, pandas as pd
import numpy as np

# ─── Configuration ──────────────────────────────────────────────────────────
DB_PATH   = Path("NVDA_put.db")     
OUT_TABLE = "daily_pnl_puts"         
CSV_FILE  = "daily_pnl_puts.csv"     
TIMEOUT   = 60                       

# ─── P&L helper ─────────────────────────────────────────────────────────────
def compute_pnl(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("V_time").copy()
    df["trade_date"] = df["V_time"].dt.normalize()              
    df["open_ask"]   = df.groupby("trade_date")["P_ask"].transform("first")
    df["pnl_tick"]   = df["P_bid"] - df["open_ask"]            

    daily = (
        df.groupby("trade_date")
          .agg(open_ask  = ("P_ask", "first"),
               close_bid = ("P_bid", "last"))
          .assign(pnl_eod = lambda x: np.floor((x.close_bid - x.open_ask) * 100) / 100)
          .loc[:, ["pnl_eod"]]                                  
          .reset_index()
    )
    return daily

with sqlite3.connect(DB_PATH, timeout=TIMEOUT) as con:
    try:
        con.execute("PRAGMA journal_mode=WAL;")
    except sqlite3.OperationalError as e:
        if "locked" in str(e).lower():
            print("[warn] Could not enable WAL; using default journal mode.")
        else:
            raise

    tables = pd.read_sql(
        """SELECT name FROM sqlite_master
           WHERE type='table' AND name NOT LIKE 'sqlite_%';""",
        con
    )["name"].tolist()

    daily_frames = []
    for tbl in tables:
        try:
            raw = pd.read_sql(
                f"""SELECT V_time,
                           P_bid AS P_bid,
                           P_ask AS P_ask
                    FROM [{tbl}]""",
                con,
                parse_dates=["V_time"]
            )

            if raw.empty:
                print(f"[skip] {tbl}: table is empty.")
                continue

            daily = compute_pnl(raw)
            daily["contract"] = tbl
            daily_frames.append(daily)

            print(f"[done] {tbl}: {len(daily)} sessions processed.")
        except Exception as err:
            print(f"[error] {tbl}: {err}")

    if not daily_frames:
        raise RuntimeError("No data found in any table—nothing to write.")

    all_daily = (
        pd.concat(daily_frames, ignore_index=True, sort=False)
          .sort_values(["contract", "trade_date"])
    )

    con.commit()
    all_daily.to_sql(OUT_TABLE, con,
                     if_exists="replace", index=False, method="multi")

print(f"✓ Wrote {len(all_daily):,} rows to '{OUT_TABLE}' in {DB_PATH.name}")

all_daily.to_csv(CSV_FILE, index=False)
print(f"✓ Also saved CSV → {CSV_FILE}")
