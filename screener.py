"""
195-Minuten-Signal-Screener (PJM-Style) für den S&P 500.

Long-Signal: EMA 5 kreuzt EMA 20 von unten nach oben (Short: von oben nach unten),
berechnet auf 195-Min-Kerzen.
  "1. Kerze" = Kreuzung auf der letzten fertigen Kerze
  "2. Kerze" = Kreuzung auf der vorletzten Kerze (eine Kerze früher)
(RSI- und Preis-Bedingung sind optional zuschaltbar, standardmäßig AUS.)

Eine US-Sitzung (9:30-16:00 ET) hat genau 2 Kerzen à 195 Min: 09:30-12:45 und 12:45-16:00 ET.
"""
import io
import json
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ----------------------------- Einstellungen -----------------------------
RSI_LEN = 14          # RSI-Länge
RSI_MA_LEN = 14       # SMA auf den RSI
EMA_FAST = 5
EMA_SLOW = 20
USE_RSI = False       # True: zusätzlich RSI kreuzt seinen SMA in Signalrichtung
USE_PRICE = False     # True: zusätzlich Kurs kreuzt EMA20 in Signalrichtung
PRICE_CROSS = "close" # "close": Schlusskurs kreuzt EMA20
                      # "candle": Kerze (Low..High) überspannt EMA20, Close auf der Signalseite
BATCH = 100           # Ticker pro Yahoo-Download
OUT_DIR = Path("docs/data")
# -------------------------------------------------------------------------

ET = ZoneInfo("America/New_York")
BAR_MIN = 195
SECTOR_ETF = {
    "Communication Services": "XLC", "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP", "Energy": "XLE", "Financials": "XLF",
    "Health Care": "XLV", "Industrials": "XLI", "Information Technology": "XLK",
    "Materials": "XLB", "Real Estate": "XLRE", "Utilities": "XLU",
}


def get_universe() -> pd.DataFrame:
    """S&P-500-Liste von Wikipedia (Ticker, Name, Sektor)."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    df = pd.read_html(io.StringIO(html))[0]
    df = df.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"})
    df["yf"] = df["ticker"].str.replace(".", "-", regex=False)
    df["etf"] = df["sector"].map(SECTOR_ETF)
    return df[["ticker", "yf", "name", "sector", "etf"]]


def to_195(df30: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """30-Min-Kerzen -> 195-Min-Kerzen (2 pro Tag, 09:30 und 12:45 ET). Nur abgeschlossene Kerzen."""
    d = df30.copy()
    d.index = d.index.tz_convert(ET)
    d = d[(d.index.time >= dt.time(9, 30)) & (d.index.time < dt.time(16, 0))]
    split = dt.time(12, 45)
    d["date"] = d.index.date
    d["bar"] = (d.index.time >= split).astype(int)
    g = d.groupby(["date", "bar"]).agg(
        Open=("Open", "first"), High=("High", "max"),
        Low=("Low", "min"), Close=("Close", "last"),
    )
    starts = [
        pd.Timestamp(dt.datetime.combine(day, dt.time(9, 30) if bar == 0 else split), tzinfo=ET)
        for day, bar in g.index
    ]
    g.index = pd.DatetimeIndex(starts)
    g["bar_no"] = [1 if t.time() == dt.time(9, 30) else 2 for t in g.index]
    # unfertige Kerze verwerfen
    g = g[g.index + pd.Timedelta(minutes=BAR_MIN) <= now]
    return g


def rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) <= b.shift(1)) & (a > b)


def cross_down(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) >= b.shift(1)) & (a < b)


def evaluate(bars: pd.DataFrame):
    """Liefert Liste von Signalen: (signal, kerze, cross_bar_start, close_letzte_kerze).
    kerze=1: Kreuzung auf letzter Kerze, kerze=2: auf der vorletzten."""
    if len(bars) < max(RSI_LEN + RSI_MA_LEN, EMA_SLOW) + 5:
        return None
    c = bars["Close"]
    r = rsi(c, RSI_LEN)
    r_ma = r.rolling(RSI_MA_LEN).mean()
    e_f = c.ewm(span=EMA_FAST, adjust=False).mean()
    e_s = c.ewm(span=EMA_SLOW, adjust=False).mean()

    long_ = cross_up(e_f, e_s)
    short_ = cross_down(e_f, e_s)
    if USE_RSI:
        long_ &= cross_up(r, r_ma)
        short_ &= cross_down(r, r_ma)
    if USE_PRICE:
        long_ &= cross_up(c, e_s)
        short_ &= cross_down(c, e_s)

    # neueste Kreuzung gewinnt (1. Kerze vor 2. Kerze)
    for kerze in (1, 2):
        i = -kerze
        if long_.iloc[i]:
            return "Long", kerze, bars.index[i], float(c.iloc[-1])
        if short_.iloc[i]:
            return "Short", kerze, bars.index[i], float(c.iloc[-1])
    return None


def main():
    now = pd.Timestamp.now(tz=ET)
    uni = get_universe()
    results, last_bar = [], None

    for i in range(0, len(uni), BATCH):
        chunk = uni.iloc[i:i + BATCH]
        raw = yf.download(
            list(chunk["yf"]), period="60d", interval="30m",
            group_by="ticker", threads=True, progress=False, auto_adjust=False,
        )
        for _, row in chunk.iterrows():
            try:
                df = raw[row["yf"]].dropna(subset=["Close"])
                bars = to_195(df, now)
                if bars.empty:
                    continue
                if last_bar is None or bars.index[-1] > last_bar:
                    last_bar = bars.index[-1]
                res = evaluate(bars)
                if res:
                    sig, kerze, t, price = res
                    results.append({
                        "signal": sig, "kerze": kerze, "ticker": row["ticker"],
                        "name": row["name"], "sector": row["sector"], "etf": row["etf"],
                        "price": round(price, 2), "cross_bar": t.isoformat(),
                        "last_bar": bars.index[-1].isoformat(),
                    })
            except Exception as e:  # einzelne Ticker dürfen scheitern
                print("skip", row["ticker"], e)

    # Ticker mit veralteten Daten (letzte Kerze != jüngste Kerze) aussortieren
    if last_bar is not None:
        results = [r for r in results if r["last_bar"] == last_bar.isoformat()]
    results.sort(key=lambda r: (r["kerze"], r["signal"], r["ticker"]))

    out = {
        "generated": now.isoformat(),
        "bar_start": last_bar.isoformat() if last_bar is not None else None,
        "settings": {"rsi": RSI_LEN, "rsi_ma": RSI_MA_LEN, "ema_fast": EMA_FAST,
                     "ema_slow": EMA_SLOW, "use_rsi": USE_RSI, "use_price": USE_PRICE},
        "signals": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "latest.json").write_text(json.dumps(out, indent=2))
    if last_bar is not None:
        arch = OUT_DIR / "archive"
        arch.mkdir(exist_ok=True)
        (arch / f"{last_bar:%Y-%m-%d}_k{(1 if last_bar.time() == dt.time(9, 30) else 2)}.json").write_text(json.dumps(out, indent=2))
    print(f"{len(results)} Signale für Kerze {out['bar_start']}")


if __name__ == "__main__":
    main()
