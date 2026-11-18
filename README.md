# 195-Min Signal-Screener

Signal: EMA 5 kreuzt EMA 20 auf der 195-Min-Kerze (Long = von unten nach oben, Short = von oben nach unten). ☀ = Signal auf der 1. Kerze des Tages, ☾ = 2. Kerze. RSI-/Preis-Filter optional (`USE_RSI`, `USE_PRICE`).
Zusätzlich gibt es Tagessignale (Button „1 Tag“): EMA 5 kreuzt EMA 20 auf der abgeschlossenen Tageskerze. Eine US-Sitzung hat 2 Kerzen (09:30-12:45 und 12:45-16:00 ET). Universum: S&P 500. Daten: Yahoo Finance (yfinance).

## Einrichten
1. Neues GitHub-Repo anlegen, alle Dateien hochladen.
2. Settings > Pages > Source: "Deploy from a branch", Branch `main`, Ordner `/docs`.
3. Actions > "Screener" > "Run workflow" (einmal manuell starten).
4. Die Seite ist dann unter https://DEINNAME.github.io/REPONAME/ erreichbar.

Parameter (EMA-Längen, USE_RSI, USE_PRICE) stehen oben in `screener.py`.
Lokal testen: `pip install -r requirements.txt && python screener.py`, dann `docs/` per `python -m http.server` öffnen.
