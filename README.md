# 195-Min Signal-Screener

Long: EMA 5 kreuzt EMA 20 von unten nach oben auf der letzten (1. Kerze) oder vorletzten (2. Kerze) 195-Min-Kerze. Short: von oben nach unten. RSI-/Preis-Filter sind optional (`USE_RSI`, `USE_PRICE`).
Eine US-Sitzung hat 2 Kerzen (09:30-12:45 und 12:45-16:00 ET). Universum: S&P 500. Daten: Yahoo Finance (yfinance).

## Einrichten
1. Neues GitHub-Repo anlegen, alle Dateien hochladen.
2. Settings > Pages > Source: "Deploy from a branch", Branch `main`, Ordner `/docs`.
3. Actions > "Screener" > "Run workflow" (einmal manuell starten).
4. Die Seite ist dann unter https://DEINNAME.github.io/REPONAME/ erreichbar.

Parameter (EMA-Längen, USE_RSI, USE_PRICE) stehen oben in `screener.py`.
Lokal testen: `pip install -r requirements.txt && python screener.py`, dann `docs/` per `python -m http.server` öffnen.
