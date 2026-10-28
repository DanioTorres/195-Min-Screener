"""
195-Minuten-Signal-Screener (PJM-Style) für den S&P 500.

Signal: EMA 5 kreuzt EMA 20 auf einer 195-Min-Kerze
  Long  = von unten nach oben,  Short = von oben nach unten
Eine US-Sitzung (9:30-16:00 ET) hat 2 Kerzen à 195 Min:
  1. Kerze (Sonne) 09:30-12:45 ET   |   2. Kerze (Mond) 12:45-16:00 ET
Gespeichert wird jedes Signal mit Datum + Kerze, damit man im Frontend
jeden Tag auswählen kann. (RSI-/Preis-Filter optional, standardmäßig AUS.)
"""
import io
import json
import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from urllib.parse import urlparse

# ----------------------------- Einstellungen -----------------------------
RSI_LEN = 14
RSI_MA_LEN = 14
EMA_FAST = 5
EMA_SLOW = 20
USE_RSI = False       # True: zusätzlich RSI kreuzt seinen SMA in Signalrichtung
USE_PRICE = False     # True: zusätzlich Close kreuzt EMA20 in Signalrichtung
WARMUP_BARS = 25      # erste Kerzen nur zum Einschwingen der Indikatoren
KEEP_DAYS = 120       # so lange bleiben alte Signale im Archiv
CAP_REFRESH_DAYS = 7  # Marktkapitalisierung nur wöchentlich neu laden
BATCH = 100
OUT_DIR = Path("docs/data")
# -------------------------------------------------------------------------

ET = ZoneInfo("America/New_York")
BAR_MIN = 195
SECTOR_RENAME = {"Information Technology": "Technology"}
SECTOR_ETF = {
    "Communication Services": "XLC", "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP", "Energy": "XLE", "Financials": "XLF",
    "Health Care": "XLV", "Industrials": "XLI", "Materials": "XLB",
    "Real Estate": "XLRE", "Technology": "XLK", "Utilities": "XLU",
}


def get_universe() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    df = pd.read_html(io.StringIO(html))[0]
    df = df.rename(columns={"Symbol": "ticker", "Security": "name",
                            "GICS Sector": "sector", "GICS Sub-Industry": "industry"})
    df["yf"] = df["ticker"].str.replace(".", "-", regex=False)
    df["sector"] = df["sector"].replace(SECTOR_RENAME)
    df["etf"] = df["sector"].map(SECTOR_ETF)
    return df[["ticker", "yf", "name", "sector", "etf", "industry"]]


def to_195(df30: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """30-Min -> 195-Min-Kerzen (2 pro Tag). Nur abgeschlossene Kerzen."""
    d = df30.copy()
    d.index = d.index.tz_convert(ET)
    d = d[(d.index.time >= dt.time(9, 30)) & (d.index.time < dt.time(16, 0))]
    split = dt.time(12, 45)
    d["date"] = d.index.date
    d["bar"] = (d.index.time >= split).astype(int)
    g = d.groupby(["date", "bar"]).agg(
        Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"),
        Close=("Close", "last"), Volume=("Volume", "sum"),
    )
    starts = [pd.Timestamp(dt.datetime.combine(day, dt.time(9, 30) if bar == 0 else split), tz=ET)
              for day, bar in g.index]
    g.index = pd.DatetimeIndex(starts)
    g["bar_no"] = [1 if t.time() == dt.time(9, 30) else 2 for t in g.index]
    return g[g.index + pd.Timedelta(minutes=BAR_MIN) <= now]


def rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def cross_up(a, b):
    return (a.shift(1) <= b.shift(1)) & (a > b)


def cross_down(a, b):
    return (a.shift(1) >= b.shift(1)) & (a < b)


def signals_for(ticker: str, bars: pd.DataFrame) -> list:
    """Alle Signale im Verlauf: [ticker, datum, kerze(1/2), 'L'/'S', kurs, tagesvolumen]."""
    if len(bars) < WARMUP_BARS + 2:
        return []
    c = bars["Close"]
    e_f = c.ewm(span=EMA_FAST, adjust=False).mean()
    e_s = c.ewm(span=EMA_SLOW, adjust=False).mean()
    long_, short_ = cross_up(e_f, e_s), cross_down(e_f, e_s)
    if USE_RSI:
        r = rsi(c, RSI_LEN)
        r_ma = r.rolling(RSI_MA_LEN).mean()
        long_ &= cross_up(r, r_ma)
        short_ &= cross_down(r, r_ma)
    if USE_PRICE:
        long_ &= cross_up(c, e_s)
        short_ &= cross_down(c, e_s)

    dates = np.array([t.date() for t in bars.index])
    day_vol = bars["Volume"].groupby(dates).transform("sum")
    rows = []
    for i in np.flatnonzero((long_ | short_).to_numpy()):
        if i < WARMUP_BARS:
            continue
        rows.append([ticker, bars.index[i].strftime("%Y-%m-%d"), int(bars["bar_no"].iloc[i]),
                     "L" if long_.iloc[i] else "S", round(float(c.iloc[i]), 2), int(day_vol.iloc[i])])
    return rows


def fetch_cap(sym: str):
    try:
        fi = yf.Ticker(sym).fast_info
        v = getattr(fi, "market_cap", None)
        if v is None:
            v = fi["marketCap"]
        return float(v) if v else None
    except Exception:
        return None


def load_caps(uni: pd.DataFrame, today: dt.date) -> dict:
    f = OUT_DIR / "caps.json"
    old = json.loads(f.read_text()) if f.exists() else {"updated": "1970-01-01", "caps": {}}
    age = (today - dt.date.fromisoformat(old["updated"])).days
    if age < CAP_REFRESH_DAYS and all(t in old["caps"] for t in uni["ticker"]):
        return old["caps"]
    print("lade Marktkapitalisierung ...")
    with ThreadPoolExecutor(16) as ex:
        vals = list(ex.map(fetch_cap, uni["yf"]))
    caps = {t: (v if v else old["caps"].get(t)) for t, v in zip(uni["ticker"], vals)}
    f.write_text(json.dumps({"updated": today.isoformat(), "caps": caps}))
    return caps


def fetch_domain(sym: str) -> str:
    try:
        web = yf.Ticker(sym).info.get("website") or ""
        host = urlparse(web if "//" in web else "//" + web).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def load_domains(sub: pd.DataFrame) -> dict:
    """Firmen-Domains (für Logo-Fallback). Wird einmalig geholt und gecacht."""
    f = OUT_DIR / "domains.json"
    doms = json.loads(f.read_text()) if f.exists() else {}
    todo = sub[~sub["ticker"].isin(doms)]
    if len(todo):
        print(f"lade Domains für {len(todo)} Aktien ...")
        with ThreadPoolExecutor(8) as ex:
            for t, d in zip(todo["ticker"], ex.map(fetch_domain, todo["yf"])):
                doms[t] = d
        f.write_text(json.dumps(doms))
    return doms


def main():
    now = pd.Timestamp.now(tz=ET)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    uni = get_universe()
    new_rows, seen_dates, last_bar = [], set(), None

    for i in range(0, len(uni), BATCH):
        chunk = uni.iloc[i:i + BATCH]
        raw = yf.download(list(chunk["yf"]), period="60d", interval="30m",
                          group_by="ticker", threads=True, progress=False, auto_adjust=False)
        for _, row in chunk.iterrows():
            try:
                df = raw[row["yf"]].dropna(subset=["Close"])
                bars = to_195(df, now)
                if bars.empty:
                    continue
                seen_dates.update(t.strftime("%Y-%m-%d") for t in bars.index[WARMUP_BARS:])
                if last_bar is None or bars.index[-1] > last_bar:
                    last_bar = bars.index[-1]
                new_rows += signals_for(row["ticker"], bars)
            except Exception as e:
                print("skip", row["ticker"], e)

    # alte Signale behalten (außer an Tagen, die gerade neu berechnet wurden)
    f = OUT_DIR / "signals.json"
    old_rows = json.loads(f.read_text())["signals"] if f.exists() else []
    cutoff = (now - pd.Timedelta(days=KEEP_DAYS)).strftime("%Y-%m-%d")
    kept = [r for r in old_rows if r[1] not in seen_dates and r[1] >= cutoff]
    rows = sorted(kept + new_rows, key=lambda r: (r[1], r[2], r[0]))

    caps = load_caps(uni, now.date())
    used = {r[0] for r in rows}
    doms = load_domains(uni[uni["ticker"].isin(used)])
    stocks = {r.ticker: {"n": r["name"], "s": r["sector"], "e": r["etf"], "i": r["industry"],
                         "mc": caps.get(r["ticker"]), "d": doms.get(r["ticker"]) or None}
              for _, r in uni.iterrows() if r["ticker"] in used}
    for t, v in stocks.items():  # NaN -> None
        v["e"] = v["e"] if isinstance(v["e"], str) else None

    out = {"generated": now.isoformat(),
           "last_bar": last_bar.isoformat() if last_bar is not None else None,
           "settings": {"ema_fast": EMA_FAST, "ema_slow": EMA_SLOW,
                        "use_rsi": USE_RSI, "use_price": USE_PRICE},
           "stocks": stocks, "signals": rows}
    f.write_text(json.dumps(out, separators=(",", ":")))
    print(f"{len(rows)} Signale, {len(stocks)} Aktien, letzte Kerze {out['last_bar']}")


if __name__ == "__main__":
    main()
