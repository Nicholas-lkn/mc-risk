"""Download daily adjusted closes for the risk layer into data/prices.csv.

Usage: uv run --extra data python scripts/download_data.py
Then:  uv run python scripts/validate.py
"""

from __future__ import annotations

from pathlib import Path

import yfinance as yf

from mc_risk.risk.data import ALL_TICKERS

OUT = Path(__file__).resolve().parents[1] / "data" / "prices.csv"


def main() -> None:
    frames = {}
    for ticker in ALL_TICKERS:  # one request per ticker: a failed symbol is obvious
        close = yf.download(ticker, start="2020-01-01", auto_adjust=True, progress=False)["Close"]
        frames[ticker] = close.squeeze()
        print(f"{ticker:8s} {frames[ticker].notna().sum()} rows")
    import pandas as pd

    prices = pd.DataFrame(frames)
    empty = [t for t in ALL_TICKERS if prices[t].notna().sum() == 0]
    if empty:
        raise SystemExit(f"no data returned for {empty}; rerun")
    OUT.parent.mkdir(exist_ok=True)
    prices.to_csv(OUT, index_label="Date")
    print(f"wrote {OUT} {prices.shape}")


if __name__ == "__main__":
    main()
