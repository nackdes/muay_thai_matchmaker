# Muay Thai Matchmaker

Eine kleine lokale Streamlit-App für Matchmaking bei Muay-Thai-Wettkämpfen.

## Funktionen

- Excel-Import per `.xlsx`
- Pflichtspalten: `Name`, `Verein`, `Geschlecht`, `Gewicht`, `Kämpfe`, `Siege`, `Niederlagen`
- Matching nach Gewicht, Anzahl Kämpfe, Siegen, Niederlagen und Siegquote
- Kämpfer aus demselben Verein werden immer ausgeschlossen
- Optional: nur gleiches Geschlecht matchen
- Globale Optimierung: maximiert zuerst die Anzahl der Paarungen und danach den Match-Score
- Export der Ergebnisse als Excel-Datei

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Unter Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Excel-Format

Die App erkennt auch einige alternative Spaltennamen, empfohlen sind aber exakt diese Spalten:

| Name | Verein | Geschlecht | Gewicht | Kämpfe | Siege | Niederlagen |
|---|---|---|---:|---:|---:|---:|
| Kämpfer 1 | Gym Alpha | männlich | 70.0 | 3 | 2 | 1 |
| Kämpfer 2 | Gym Beta | männlich | 71.2 | 4 | 2 | 2 |

## Matching-Score

Der Score liegt zwischen 0 und 100. Je näher zwei Kämpfer bei Gewicht, Kampferfahrung, Siegen, Niederlagen und Siegquote liegen, desto höher ist der Score. Gleiche Vereine werden vor der Score-Berechnung ausgeschlossen.

## Datenschutz

Die App läuft lokal auf deinem Rechner. Die hochgeladene Excel-Datei wird nicht an einen Server übertragen, außer du hostest die App selbst auf einer externen Plattform.
