# 🥊 Muay Thai & K1 Matchmaker Pro

Ein intelligentes, graphbasiertes Turniermanagement-Tool, das auf Basis von Excel-Meldelisten vollautomatisch optimale und faire Kampfpaarungen (Matches) generiert. Die App stellt sicher, dass keine Person doppelt gematcht wird und schließt Vereinskollegen kategorisch als Gegner aus.

---

## ✨ Features (Neueste Version)

- **Regelwerk-Trennung (Muay Thai / K-1):** Kämpfer werden anhand ihrer gemeldeten Disziplin unterschieden. Die App trennt die Kämpfe standardmäßig strikt, erlaubt bei Bedarf aber auch stilübergreifende Paarungen.
- **Dynamische Gewichtsklassen:** Über die Seitenleiste können Gewichtsklassen live eingesehen, gelöscht oder neue Obergrenzen flexibel hinzugefügt werden (z. B. für Jugend- oder Superschwergewichte).
- **Alters- & Erfahrungs-Validierung:** Berücksichtigt das Alter sowie die Kampfanzahl (Einteilung in Newcomer, D-, C-, B- und A-Klasse).
- **Einstellbare Gewichtung (Strafen-Multiplikator):** Bestimme selbst, welche Kriterien (z. B. Gewicht vor Erfahrung) dem Algorithmus bei Abweichungen am wichtigsten sind.
- **Qualitätsfilter (Mindest-Score):** Verhindert unfaire Kämpfe im Gesamtpaket, selbst wenn einzelne Grenzwerte knapp eingehalten wurden.
- **🛠️ Manueller Matchmaker:** Nachzügler oder schwer vermittelbare Kämpfer aus der "Ungematcht"-Liste können am Ende manuell zu Paarungen zusammengefügt werden.
- **Excel-Export & Vorlage:** Direkter Download einer passenden Import-Vorlage und Export der finalen Kampflisten (inklusive mathematischer Auswertungen) als Excel-Datei.

---

## 📋 Anforderungen an die Excel-Datei

Die hochgeladene Excel-Datei muss eine Tabelle enthalten, bei der folgende **Pflichtspalten** vorhanden sein müssen (Groß-/Kleinschreibung und gängige Synonyme wie "Gym" statt "Verein" oder "Fights" statt "Kämpfe" werden automatisch erkannt):

1. `Name`
2. `Verein`
3. `Geschlecht` (männlich / weiblich / divers)
4. `Alter`
5. `Gewicht` (in kg, z. B. `71.5`)
6. `Disziplin` (Muay Thai / K-1)
7. `Kämpfe` (Gesamtanzahl)
8. `Siege`
9. `Niederlagen`

*Tipp: Nutze den Button "Excel-Vorlage herunterladen" in der App, um eine perfekt formatierte Startdatei zu erhalten.*

---

## 🚀 Installation & Lokaler Start

### 1. Repository klonen oder Dateien in einen Ordner kopieren
Stelle sicher, dass sich `app.py`, `requirements.txt` und diese `README.md` im selben Verzeichnis befinden.

### 2. Virtuelle Umgebung erstellen (Empfohlen)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate