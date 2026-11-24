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
import os
import sys
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FTimeout
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
WARMUP_BARS = 60      # erste Kerzen nur zum Einschwingen der EMAs (TradingView rechnet mit längerer Historie)
KEEP_DAYS = 120       # so lange bleiben alte Signale im Archiv
CAP_REFRESH_DAYS = 7  # Marktkapitalisierung nur wöchentlich neu laden
CAP_BUDGET_S = 150    # max. Sekunden fürs Nachladen der Marktkapitalisierung
DOMAIN_BUDGET_S = 150 # max. Sekunden fürs Nachladen der Firmen-Domains
DOMAIN_MAX_PER_RUN = 120  # pro Lauf höchstens so viele neue Domains
DAILY_KEEP_DAYS = 120     # so viele Kalendertage Tagessignale werden angezeigt
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
    """15-Min -> 195-Min-Kerzen (2 pro Tag, Schnitt exakt um 12:45 ET). Nur abgeschlossene Kerzen."""
    d = df30.copy()
    d.index = d.index.tz_convert(ET)
    d = d[(d.index.time >= dt.time(9, 30)) & (d.index.time < dt.time(16, 0))]
    split = dt.time(12, 45)
    d["date"] = d.index.date
    d["bar"] = (d.index.time >= split).astype(int)
    d["ts"] = d.index
    g = d.groupby(["date", "bar"]).agg(
        Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"),
        Close=("Close", "last"), Volume=("Volume", "sum"), last_t=("ts", "last"),
    )
    starts = [pd.Timestamp(dt.datetime.combine(day, dt.time(9, 30) if bar == 0 else split), tz=ET)
              for day, bar in g.index]
    g.index = pd.DatetimeIndex(starts)
    g["bar_no"] = [1 if t.time() == dt.time(9, 30) else 2 for t in g.index]
    g = g[g.index + pd.Timedelta(minutes=BAR_MIN) <= now]
    # Kerze nur gelten lassen, wenn ihre letzte 15-Min-Kerze (12:30 bzw. 15:45) wirklich vorliegt.
    # Sonst wäre der Schlusskurs unvollständig und das Signal könnte sich später ändern.
    need = g["bar_no"].map({1: dt.time(12, 30), 2: dt.time(15, 45)})
    ok = [t.time() >= n for t, n in zip(g["last_t"], need)]
    return g[ok]


def rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def cross_up(a, b):
    return (a.shift(1) <= b.shift(1)) & (a > b)


def cross_down(a, b):
    return (a.shift(1) >= b.shift(1)) & (a < b)


def cross_masks(c: pd.Series):
    """Long-/Short-Kreuzung (EMA 5 kreuzt EMA 20), optional mit RSI-/Preis-Filter."""
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
    return long_, short_


def signals_for(ticker: str, bars: pd.DataFrame) -> list:
    """195-Min: [ticker, datum, kerze(1/2), 'L'/'S', kurs, tagesvolumen]."""
    if len(bars) < WARMUP_BARS + 2:
        return []
    c = bars["Close"]
    long_, short_ = cross_masks(c)
    dates = np.array([t.date() for t in bars.index])
    day_vol = bars["Volume"].groupby(dates).transform("sum")
    rows = []
    for i in np.flatnonzero((long_ | short_).to_numpy()):
        if i < WARMUP_BARS:
            continue
        rows.append([ticker, bars.index[i].strftime("%Y-%m-%d"), int(bars["bar_no"].iloc[i]),
                     "L" if long_.iloc[i] else "S", round(float(c.iloc[i]), 2), int(day_vol.iloc[i])])
    return rows


def daily_signals(ticker: str, df: pd.DataFrame, now: pd.Timestamp, cutoff: str) -> list:
    """Tagessignale: [ticker, datum, 0, 'L'/'S', kurs, volumen]. Nur abgeschlossene Tageskerzen."""
    df = df.dropna(subset=["Close"])
    dates = [t.date() for t in df.index]
    today = now.date()
    done = [d < today or (d == today and now.time() >= dt.time(16, 5)) for d in dates]
    df = df[done]
    if len(df) < WARMUP_BARS + 2:
        return []
    c = df["Close"]
    long_, short_ = cross_masks(c)
    rows = []
    for i in np.flatnonzero((long_ | short_).to_numpy()):
        if i < WARMUP_BARS:
            continue
        d = df.index[i].strftime("%Y-%m-%d")
        if d < cutoff:
            continue
        vol = df["Volume"].iloc[i]
        rows.append([ticker, d, 0, "L" if long_.iloc[i] else "S", round(float(c.iloc[i]), 2),
                     int(vol) if pd.notna(vol) else 0])
    return rows


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def pool_map(fn, items, workers, budget):
    """Führt fn parallel aus, bricht nach `budget` Sekunden ab (hängende Anfragen blockieren nichts)."""
    ex = ThreadPoolExecutor(workers)
    futs = {ex.submit(fn, x): x for x in items}
    out = {}
    try:
        for f in as_completed(futs, timeout=budget):
            try:
                out[futs[f]] = f.result()
            except Exception:
                pass
    except FTimeout:
        print(f"Zeitbudget ({budget}s) erreicht, {len(out)}/{len(items)} fertig")
    ex.shutdown(wait=False, cancel_futures=True)
    return out


def fetch_cap(sym: str):
    try:
        fi = yf.Ticker(sym).fast_info
        v = getattr(fi, "market_cap", None)
        if v is None:
            v = fi["marketCap"]
        return float(v) if v else None
    except Exception:
        return None


def fetch_domain(sym: str):
    """Domain der Firmen-Website; None bei Fehler (wird dann später erneut versucht)."""
    try:
        web = yf.Ticker(sym).info.get("website") or ""
        host = urlparse(web if "//" in web else "//" + web).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return None


def main():
    now = pd.Timestamp.now(tz=ET)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    uni = get_universe()
    new_rows, daily_rows, seen_dates, last_bar, got = [], [], set(), None, 0
    d_cutoff = (now - pd.Timedelta(days=DAILY_KEEP_DAYS)).strftime("%Y-%m-%d")

    for i in range(0, len(uni), BATCH):
        chunk = uni.iloc[i:i + BATCH]
        raw = yf.download(list(chunk["yf"]), period="60d", interval="15m",
                          group_by="ticker", threads=True, progress=False, auto_adjust=False, timeout=30)
        try:
            rawd = yf.download(list(chunk["yf"]), period="1y", interval="1d",
                               group_by="ticker", threads=True, progress=False, auto_adjust=False, timeout=30)
        except Exception as e:
            print("Tagesdaten fehlgeschlagen", e)
            rawd = None
        for _, row in chunk.iterrows():
            try:
                if rawd is not None:
                    daily_rows += daily_signals(row["ticker"], rawd[row["yf"]], now, d_cutoff)
            except Exception as e:
                print("skip daily", row["ticker"], e)
            try:
                df = raw[row["yf"]].dropna(subset=["Close"])
                got += 1 if len(df) else 0
                bars = to_195(df, now)
                if bars.empty:
                    continue
                seen_dates.update(t.strftime("%Y-%m-%d") for t in bars.index[WARMUP_BARS:])
                if last_bar is None or bars.index[-1] > last_bar:
                    last_bar = bars.index[-1]
                new_rows += signals_for(row["ticker"], bars)
            except Exception as e:
                print("skip", row["ticker"], e)

    print(f"Kursdaten für {got} von {len(uni)} Aktien erhalten", flush=True)
    if got < len(uni) * 0.5:
        print("FEHLER: Yahoo hat kaum Daten geliefert - bestehende Signale bleiben unverändert.", flush=True)
        sys.stdout.flush()
        os._exit(1)

    # alte Signale behalten (außer an Tagen, die gerade neu berechnet wurden)
    f = OUT_DIR / "signals.json"
    old_rows = read_json(f, {"signals": []})["signals"]
    cutoff = (now - pd.Timedelta(days=KEEP_DAYS)).strftime("%Y-%m-%d")
    kept = [r for r in old_rows if r[1] not in seen_dates and r[1] >= cutoff]
    rows = sorted(kept + new_rows, key=lambda r: (r[1], r[2], r[0]))

    daily_rows.sort(key=lambda r: (r[1], r[0]))
    used = {r[0] for r in rows} | {r[0] for r in daily_rows}
    sub = uni[uni["ticker"].isin(used)]
    capf = read_json(OUT_DIR / "caps.json", {"updated": "1970-01-01", "caps": {}})
    caps = capf["caps"]
    doms = read_json(OUT_DIR / "domains.json", {})

    def write():
        stocks = {}
        for _, r in sub.iterrows():
            stocks[r["ticker"]] = {"n": r["name"], "s": r["sector"],
                                   "e": r["etf"] if isinstance(r["etf"], str) else None,
                                   "i": r["industry"], "mc": caps.get(r["ticker"]),
                                   "d": doms.get(r["ticker"]) or None}
        out = {"generated": now.isoformat(),
               "last_bar": last_bar.isoformat() if last_bar is not None else None,
               "settings": {"ema_fast": EMA_FAST, "ema_slow": EMA_SLOW,
                            "use_rsi": USE_RSI, "use_price": USE_PRICE},
               "stocks": stocks, "signals": rows, "daily": daily_rows}
        f.write_text(json.dumps(out, separators=(",", ":")))

    write()   # Signale sind jetzt sicher gespeichert, auch wenn das Nachladen unten scheitert
    print(f"{len(rows)} 195-Min-Signale, {len(daily_rows)} Tagessignale gespeichert, letzte Kerze {last_bar}", flush=True)

    # --- Marktkapitalisierung nachladen (mit Zeitbudget) ---
    age = (now.date() - dt.date.fromisoformat(capf["updated"])).days
    todo = [(t, y) for t, y in zip(sub["ticker"], sub["yf"]) if age >= CAP_REFRESH_DAYS or caps.get(t) is None]
    if todo:
        res = pool_map(lambda x: fetch_cap(x[1]), todo, 12, CAP_BUDGET_S)
        for (t, _), v in res.items():
            if v:
                caps[t] = v
        (OUT_DIR / "caps.json").write_text(json.dumps({"updated": now.date().isoformat(), "caps": caps}))
        write()

    # --- Firmen-Domains für Logos nachladen (nur Neue, mit Zeitbudget) ---
    todo = [(t, y) for t, y in zip(sub["ticker"], sub["yf"]) if t not in doms][:DOMAIN_MAX_PER_RUN]
    if todo:
        res = pool_map(lambda x: fetch_domain(x[1]), todo, 6, DOMAIN_BUDGET_S)
        for (t, _), d in res.items():
            if d is not None:
                doms[t] = d
        (OUT_DIR / "domains.json").write_text(json.dumps(doms))
        write()
    print("fertig", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)   # beendet auch dann sauber, wenn noch eine Yahoo-Anfrage im Hintergrund hängt
