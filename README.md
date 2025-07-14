# 🔍 Offenheits-Suche für Organisationen

Ein Python-Tool zur automatisierten Analyse von Organisationen hinsichtlich ihrer Offenheitskriterien mit moderner Streamlit-Oberfläche.

## ✨ Features

- 📋 **Kriterienkataloge**: Laden von YAML-Kriterienkatalogen aus dem `/criteria` Verzeichnis
- 🏢 **Organisationen**: Import von Organisationen aus CSV-Dateien (Format: Name;URL)
- 🔍 **Flexible Auswahl**: Alle Organisationen, Bereich (X bis Y) oder einzelne Organisationen
- 🤖 **Zwei Analyse-Modi**: 
  1. **Gemini CLI** (Standard): Gemini 2.5-Flash über Kommandozeile für Datensammlung + OpenAI für strukturierte Bewertung
  2. **DuckDuckGo + OpenAI** (Alternative): Web-Suche mit Proxy-Rotation + direkte OpenAI-Analyse
- 🌐 **Intelligente Web-Suche**: DuckDuckGo-Integration mit automatischer Proxy-Rotation für zuverlässige Suchergebnisse
- 🕷️ **Web-Crawling**: Automatisches Crawling aller gefundenen URLs für umfassende Datensammlung
- 📊 **Bewertungsskalen**: 3-Punkt-Likert (vorhanden/teilweise vorhanden/nicht vorhanden)
- 📈 **Erweiterte Visualisierungen**: 
  - Top/Flop-Kriterien Charts
  - Organisationsvergleiche
  - Heatmaps (Organisation vs Kriterien)
  - Verteilungsdiagramme
  - Alle als hochauflösende PNG-Dateien exportierbar
- 📥 **Umfassender Export**: 
  - Hauptergebnisse CSV
  - Kriterienanalyse CSV
  - Organisationszusammenfassung CSV
  - Detaillierte Markdown-Berichte pro Organisation
- ⚡ **Automatisches Speichern**: Alle Ergebnisse mit Zeitstempel
- 🎯 **Rate Limiting**: Intelligente Ratenbegrenzung für API-Aufrufe
- 🔄 **Proxy-Rotation**: Automatische Proxy-Verwaltung für stabile Web-Zugriffe

## 🛠️ Voraussetzungen

### System-Anforderungen
- **Python 3.8+**
- **Node.js 20+** (für Gemini CLI)
- **Internetverbindung** für API-Aufrufe

### API-Schlüssel
- **OpenAI API Key** ([hier erhalten](https://platform.openai.com/api-keys))
- **Google AI Studio API Key** für Gemini ([hier erhalten](https://aistudio.google.com/app/apikey))

## 📦 Installation

### 1. Repository klonen
```bash
git clone <repository-url>
cd offenheitscrawler2
```

### 2. Python-Abhängigkeiten installieren

**Empfohlen: Virtuelle Umgebung erstellen**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

**Abhängigkeiten installieren:**
```bash
pip install -r requirements.txt
```

**Wichtige Pakete in requirements.txt:**
- `streamlit` - Web-Interface
- `openai` - OpenAI API Integration
- `ddgs` - DuckDuckGo Suche
- `requests` - HTTP-Requests für Web-Crawling
- `beautifulsoup4` - HTML-Parsing
- `plotly` - Visualisierungen
- `pandas` - Datenverarbeitung
- `pyyaml` - YAML-Dateien lesen
- `loguru` - Erweiterte Logging-Funktionen

### 3. Gemini CLI installieren (für Gemini-Modus)

**Schritt 1: Node.js installieren**
- **Windows**: [nodejs.org](https://nodejs.org/) - LTS Version herunterladen
- **Linux**: `sudo apt install nodejs npm` oder `sudo yum install nodejs npm`
- **macOS**: `brew install node` oder von [nodejs.org](https://nodejs.org/)

**Node.js Version prüfen:**
```bash
node --version  # Sollte v20+ sein
npm --version   # Sollte v9+ sein
```

**Schritt 2: Gemini CLI installieren**

**Option A: Globale Installation (empfohlen)**
```bash
npm install -g @google/gemini-cli
```

**Option B: Lokale Installation**
```bash
npm install @google/gemini-cli
# Dann mit npx verwenden: npx gemini
```

**Schritt 3: API-Key konfigurieren**
```bash
gemini config set apiKey YOUR_GOOGLE_AI_STUDIO_API_KEY
```

**Schritt 4: Installation testen**
```bash
gemini --version
gemini -p "Test prompt" # Kurzer Funktionstest
```

**Troubleshooting Gemini CLI:**
```bash
# Falls "gemini" Befehl nicht gefunden wird:
# Windows: PATH prüfen
where gemini

# Linux/macOS: PATH prüfen
which gemini

# NPM global Pfad anzeigen
npm config get prefix

# Gemini CLI neu installieren falls Probleme
npm uninstall -g @google/gemini-cli
npm install -g @google/gemini-cli
```

### 4. Umgebungsvariablen setzen (optional)

**Windows:**
```cmd
set OPENAI_API_KEY=your_openai_api_key_here
set GOOGLE_AI_STUDIO_API_KEY=your_gemini_api_key_here
```

**Linux/macOS:**
```bash
export OPENAI_API_KEY=your_openai_api_key_here
export GOOGLE_AI_STUDIO_API_KEY=your_gemini_api_key_here
```

## 🚀 Verwendung

### 1. Anwendung starten
```bash
streamlit run app.py
```

### 2. Web-Interface verwenden

#### 🔑 Schritt 1: API-Konfiguration
- **OpenAI API Key**: Eingeben (falls nicht als Umgebungsvariable gesetzt)
- **Modell auswählen**: 
  - `gpt-4o-mini` (Standard, kosteneffizient)
  - `gpt-4o` (Leistungsstärker, teurer)
  - `gpt-3.5-turbo` (Günstigste Option)
- **Base URL**: Standard `https://api.openai.com/v1` (anpassbar für Proxies)
- **Max Tokens**: Standard 15.000 (anpassbar je nach Bedarf)

#### 📋 Schritt 2: Kriterienkatalog auswählen
- **Verfügbare Kataloge**: Aus `/criteria` Verzeichnis wählen
  - `hochschulen.yaml` - Spezifisch für Hochschulen
  - `forschungseinrichtungen.yaml` - Für Forschungsinstitute
  - Weitere benutzerdefinierte YAML-Dateien
- **Kriterien-Vorschau**: Vollständige Anzeige aller Kriterien mit Beschreibungen
- **Gewichtung**: Automatische Berücksichtigung der in YAML definierten Gewichte

#### 🏢 Schritt 3: Organisationen laden
- **CSV-Format**: `Name;URL` (Semikolon-getrennt, UTF-8 kodiert)
- **Auswahloptionen**:
  - **Alle Organisationen**: Komplette Liste analysieren
  - **Bereich (X bis Y)**: Bestimmte Zeilen auswählen (z.B. 1-10)
  - **Einzelne Organisationen**: Spezifische Organisationen per Checkbox
- **Vorschau**: Anzeige der geladenen Organisationen vor Analyse

#### ⚙️ Schritt 4: Analyse-Modus wählen

**Option A: Gemini CLI (Standard-Modus)**
- **Voraussetzung**: Gemini CLI muss installiert und konfiguriert sein
- **Prozess**: 
  1. Gemini 2.5-Flash sammelt Daten von Organisations-Webseiten
  2. OpenAI GPT analysiert und bewertet strukturiert
- **Vorteile**: Sehr detaillierte Datensammlung, hohe Qualität
- **Rate Limiting**: 6 Sekunden zwischen Gemini-Aufrufen (10 Anfragen/Minute)
- **Timeout**: 120 Sekunden pro Gemini-Aufruf

**Option B: DuckDuckGo + OpenAI (Alternative)**
- **Voraussetzung**: Nur OpenAI API Key erforderlich
- **Prozess**:
  1. DuckDuckGo-Suche nach organisationsspezifischen Informationen
  2. Automatisches Crawling aller gefundenen URLs (bis zu 10 pro Organisation)
  3. Direkte OpenAI-Analyse der gecrawlten Inhalte
- **Vorteile**: Keine Gemini CLI Installation nötig, breitere Datenquellen
- **Proxy-Rotation**: Automatische Verwendung verschiedener Proxies für stabile Suche
- **Rate Limiting**: Intelligente Verzögerungen zwischen Anfragen

#### 📊 Schritt 5: Bewertungseinstellungen
- **Bewertungsskala**: 3-Punkt-Likert (Standard)
  - `vorhanden` (1.0) - Kriterium vollständig erfüllt
  - `teilweise vorhanden` (0.5) - Kriterium teilweise erfüllt
  - `nicht vorhanden` (0.0) - Kriterium nicht erfüllt
- **Konfidenz-Bewertung**: Automatische Einschätzung der Bewertungssicherheit
- **Quellenangaben**: Dokumentation der verwendeten Informationsquellen

#### 🚀 Schritt 6: Analyse starten
- **Fortschrittsanzeige**: 
  - Aktuell analysierte Organisation
  - Fortschritt pro Kriterium
  - Geschätzte verbleibende Zeit
  - Erfolgs-/Fehlerstatistiken
- **Live-Updates**: Echtzeitanzeige der Analyseergebnisse
- **Automatisches Speichern**: Kontinuierliche Sicherung aller Ergebnisse
- **Fehlerbehandlung**: Automatische Wiederholung bei temporären Fehlern

### 3. Ergebnisse erkunden

**5-Tab-Interface:**
- **📊 Übersicht**: Zusammenfassung und Metriken
- **📋 Detailergebnisse**: Vollständige Ergebnistabelle
- **📈 Analysetabellen**: Kriterien- und Organisationsanalyse
- **🎨 Visualisierungen**: Interaktive Charts und PNG-Downloads
- **📥 Export & Downloads**: Datei-Downloads und gespeicherte Ergebnisse

## 📁 Verzeichnisstruktur

```
offenheitscrawler2/
├── criteria/                    # YAML-Kriterienkataloge
│   ├── hochschulen.yaml
│   ├── forschungseinrichtungen.yaml
│   └── weitere_kataloge.yaml
├── data/                        # CSV-Dateien mit Organisationen
│   ├── universitaeten.csv       # Format: Name;URL
│   ├── forschungsinstitute.csv
│   └── weitere_organisationen.csv
├── results/                     # Automatisch erstellte Ergebnisse
│   ├── offenheitsanalyse_*.csv
│   ├── kriterienanalyse_*.csv
│   ├── organisationszusammenfassung_*.csv
│   ├── berichte_*/              # Markdown-Berichte pro Organisation
│   ├── top_kriterien_*.png
│   ├── flop_kriterien_*.png
│   ├── organisationen_vergleich_*.png
│   ├── heatmap_*.png
│   └── kriterienverteilung_*.png
├── app.py                       # Hauptanwendung
├── requirements.txt             # Python-Abhängigkeiten
├── README.md                    # Diese Dokumentation
└── offenheitscrawler.log       # Anwendungsprotokoll
```

## ⚙️ Konfiguration

### OpenAI-Einstellungen
- **API Key**: Aus Umgebungsvariable oder manuell in der App eingeben
- **Base URL**: Standard `https://api.openai.com/v1`
- **Verfügbare Modelle**: 
  - `gpt-4o-mini` (Standard, kosteneffizient)
  - `gpt-4o` (Leistungsstark)
  - `gpt-4-turbo` (Schnell)
  - `gpt-3.5-turbo` (Günstig)
- **Max Tokens**: Standard 15.000

### Gemini-Einstellungen
- **Modell**: `gemini-2.5-flash` (automatisch)
- **Rate Limiting**: 6 Sekunden zwischen Aufrufen (10 Anfragen/Minute)
- **Timeout**: 120 Sekunden pro Aufruf
- **Fehlerbehandlung**: Automatische Wiederholung bei Quota-Fehlern

### Bewertungsskala
**3-Punkt-Likert-Skala:**
- `vorhanden` (1.0) - Kriterium ist vollständig erfüllt
- `teilweise vorhanden` (0.5) - Kriterium ist teilweise erfüllt
- `nicht vorhanden` (0.0) - Kriterium ist nicht erfüllt

## 🔄 Workflow

1. **🔍 Datensammlung**: 
   - Für jedes Kriterium wird Gemini 2.5-Flash über CLI aufgerufen
   - Webseiten-Inhalte werden analysiert und relevante Informationen extrahiert
   - Rate Limiting verhindert API-Quota-Überschreitungen

2. **🧠 Strukturierte Analyse**: 
   - OpenAI GPT analysiert die Gemini-Antworten
   - Strukturierte Bewertungen mit Begründungen werden erstellt
   - Konfidenzwerte und Quellen werden dokumentiert

3. **📊 Datenverarbeitung**:
   - Gewichtete Bewertungen nach Kriterien-Wichtigkeit
   - Statistische Auswertungen (Mittelwerte, Standardabweichungen)
   - Ranking-Erstellung für Organisationen und Kriterien

4. **📈 Visualisierung**: 
   - Automatische Erstellung von 5 verschiedenen Diagrammtypen
   - Hochauflösende PNG-Exporte für Präsentationen
   - Interaktive Charts in der Web-App

5. **💾 Export und Speicherung**:
   - 3 verschiedene CSV-Dateien mit unterschiedlichen Aggregationsebenen
   - Detaillierte Markdown-Berichte pro Organisation
   - Zeitstempel-basierte Dateibenennung für Versionierung

## 📊 Ausgabe-Dateien

### 📄 CSV-Exporte (im `/results` Verzeichnis)

#### 1. Hauptergebnisse (`offenheitsanalyse_YYYY-MM-DD_HH-MM-SS.csv`)
**Inhalt**: Vollständige Ergebnismatrix mit allen Einzelbewertungen
**Spalten**:
- `Organisation` - Name der analysierten Organisation
- `URL` - Webseite der Organisation
- `Kriterium` - Name des bewerteten Kriteriums
- `Kategorie` - Kriterien-Kategorie (z.B. "Transparenz", "Partizipation")
- `Bewertung` - Numerische Bewertung (0.0, 0.5, 1.0)
- `Bewertung_Text` - Textuelle Bewertung ("nicht vorhanden", "teilweise vorhanden", "vorhanden")
- `Begründung` - Detaillierte Erklärung der Bewertung
- `Konfidenz` - Sicherheit der Bewertung (0.0-1.0)
- `Quellen` - Verwendete Informationsquellen
- `Gewicht` - Gewichtung des Kriteriums
- `Gewichtete_Bewertung` - Bewertung × Gewicht
**Verwendung**: Detailanalyse, Pivot-Tabellen, weitere statistische Auswertungen

#### 2. Kriterienanalyse (`kriterienanalyse_YYYY-MM-DD_HH-MM-SS.csv`)
**Inhalt**: Statistische Auswertung pro Kriterium über alle Organisationen
**Spalten**:
- `Kriterium` - Name des Kriteriums
- `Kategorie` - Kriterien-Kategorie
- `Anzahl_Organisationen` - Anzahl bewerteter Organisationen
- `Vorhanden_Anzahl` - Anzahl "vorhanden" Bewertungen
- `Teilweise_Anzahl` - Anzahl "teilweise vorhanden" Bewertungen
- `Nicht_Vorhanden_Anzahl` - Anzahl "nicht vorhanden" Bewertungen
- `Vorhanden_Prozent` - Prozentsatz "vorhanden" Bewertungen
- `Teilweise_Prozent` - Prozentsatz "teilweise vorhanden" Bewertungen
- `Nicht_Vorhanden_Prozent` - Prozentsatz "nicht vorhanden" Bewertungen
- `Mittelwert` - Durchschnittliche Bewertung (0.0-1.0)
- `Standardabweichung` - Streuung der Bewertungen
- `Gewicht` - Gewichtung des Kriteriums
- `Gewichteter_Mittelwert` - Mittelwert × Gewicht
**Verwendung**: Identifikation von Top/Flop-Kriterien, Benchmark-Analysen

#### 3. Organisationszusammenfassung (`organisationszusammenfassung_YYYY-MM-DD_HH-MM-SS.csv`)
**Inhalt**: Ranking und Gesamtbewertung aller Organisationen
**Spalten**:
- `Rang` - Position im Ranking (1 = beste Organisation)
- `Organisation` - Name der Organisation
- `URL` - Webseite der Organisation
- `Gesamtscore` - Ungewichteter Durchschnitt aller Kriterien
- `Gewichteter_Gesamtscore` - Gewichteter Durchschnitt
- `Anzahl_Kriterien` - Anzahl bewerteter Kriterien
- `Vorhanden_Anzahl` - Anzahl "vorhanden" Bewertungen
- `Teilweise_Anzahl` - Anzahl "teilweise vorhanden" Bewertungen
- `Nicht_Vorhanden_Anzahl` - Anzahl "nicht vorhanden" Bewertungen
- `Beste_Kategorie` - Kategorie mit höchster Bewertung
- `Schwächste_Kategorie` - Kategorie mit niedrigster Bewertung
- `Durchschnittliche_Konfidenz` - Mittlere Bewertungssicherheit
**Verwendung**: Organisationsranking, Vergleichsanalysen, Executive Summary

### 📈 Visualisierungen (PNG-Dateien, 300 DPI)

#### 1. Top-Kriterien (`top_kriterien_YYYY-MM-DD_HH-MM-SS.png`)
- **Inhalt**: Balkendiagramm der 10 besten Kriterien nach Durchschnittsbewertung
- **Y-Achse**: Kriterien-Namen (gekürzt)
- **X-Achse**: Durchschnittsbewertung (0.0-1.0)
- **Farben**: Grün-Gradient je nach Bewertung
- **Verwendung**: Identifikation von Stärken, Best-Practice-Bereiche

#### 2. Flop-Kriterien (`flop_kriterien_YYYY-MM-DD_HH-MM-SS.png`)
- **Inhalt**: Balkendiagramm der 10 schlechtesten Kriterien
- **Y-Achse**: Kriterien-Namen (gekürzt)
- **X-Achse**: Durchschnittsbewertung (0.0-1.0)
- **Farben**: Rot-Gradient je nach Bewertung
- **Verwendung**: Identifikation von Verbesserungspotenzialen

#### 3. Organisationsvergleich (`organisationsvergleich_YYYY-MM-DD_HH-MM-SS.png`)
- **Inhalt**: Horizontales Balkendiagramm aller Organisationen
- **Y-Achse**: Organisations-Namen (sortiert nach Score)
- **X-Achse**: Gewichteter Gesamtscore (0.0-1.0)
- **Farben**: Blau-Gradient je nach Ranking
- **Verwendung**: Gesamtranking, Leistungsvergleich

#### 4. Heatmap (`heatmap_YYYY-MM-DD_HH-MM-SS.png`)
- **Inhalt**: Matrix-Darstellung Organisation vs Kriterien
- **Y-Achse**: Organisations-Namen
- **X-Achse**: Kriterien-Namen (rotiert)
- **Farben**: Rot (0.0) über Gelb (0.5) zu Grün (1.0)
- **Verwendung**: Detailvergleich, Muster-Erkennung

#### 5. Verteilungsdiagramm (`verteilung_YYYY-MM-DD_HH-MM-SS.png`)
- **Inhalt**: Kreisdiagramm der Bewertungsverteilung
- **Segmente**: "Vorhanden", "Teilweise vorhanden", "Nicht vorhanden"
- **Farben**: Grün, Gelb, Rot
- **Prozentangaben**: Anteil jeder Bewertungskategorie
- **Verwendung**: Überblick über Gesamtverteilung

### 📝 Berichte (im `/results/berichte_YYYY-MM-DD_HH-MM-SS/` Verzeichnis)

#### Einzelberichte pro Organisation (`[Organisationsname].md`)
**Struktur**:
```markdown
# Offenheitsanalyse: [Organisationsname]

## Übersicht
- **URL**: [Organisations-URL]
- **Gesamtscore**: X.XX (Rang Y von Z)
- **Analysedatum**: YYYY-MM-DD HH:MM:SS

## Bewertung nach Kategorien
### [Kategorie 1]
- **Durchschnitt**: X.XX
- **Kriterien**: N

#### [Kriterium 1.1]
- **Bewertung**: vorhanden/teilweise vorhanden/nicht vorhanden
- **Begründung**: [Detaillierte Erklärung]
- **Konfidenz**: X.XX
- **Quellen**: [Verwendete URLs/Dokumente]

## Stärken
- [Liste der besten Kriterien]

## Verbesserungspotenziale
- [Liste der schwächsten Kriterien]

## Empfehlungen
- [Konkrete Handlungsempfehlungen]
```

#### Audit-Trail (`audit_trail.json`)
**Inhalt**: Vollständige Dokumentation aller API-Aufrufe
- Gemini-Rohantworten (falls Gemini-Modus)
- OpenAI-Prompts und -Antworten
- Zeitstempel aller Aufrufe
- Fehler und Wiederholungen
- Verwendete Parameter und Einstellungen
**Verwendung**: Nachvollziehbarkeit, Debugging, Qualitätssicherung

## 🔧 Troubleshooting

### 🚑 Installation & Setup

**Problem: Gemini CLI nicht gefunden**
```bash
# 1. Node.js Version prüfen
node --version  # Muss v20+ sein
npm --version   # Muss v9+ sein

# 2. Gemini CLI Status prüfen
which gemini    # Linux/macOS
where gemini    # Windows

# 3. NPM Global-Pfad prüfen
npm config get prefix
npm list -g @google/gemini-cli

# 4. Neuinstallation
npm uninstall -g @google/gemini-cli
npm cache clean --force
npm install -g @google/gemini-cli

# 5. Alternative: Lokale Installation
npm install @google/gemini-cli
# Dann mit npx verwenden: npx gemini
```

**Problem: Python-Pakete fehlen**
```bash
# Virtuelle Umgebung aktivieren
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Requirements neu installieren
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall

# Einzelne Pakete prüfen
pip list | grep streamlit
pip list | grep openai
```

### 🔑 API-Konfiguration

**Problem: OpenAI API-Fehler**
- **401 Unauthorized**: API-Key prüfen, eventuell neu generieren
- **429 Rate Limit**: Warten oder höheres Tier buchen
- **400 Bad Request**: Max Tokens reduzieren (Standard: 15.000)
- **Connection Error**: Base URL prüfen, Proxy-Einstellungen

**Problem: Gemini API-Fehler**
```bash
# API-Key neu setzen
gemini config set apiKey YOUR_NEW_API_KEY

# Konfiguration prüfen
gemini config list

# Test-Aufruf
gemini -p "Hello World" --timeout 30
```

### 🌐 DuckDuckGo + OpenAI Modus

**Problem: Keine Suchergebnisse**
- **Lösung**: Proxy-Rotation ist automatisch aktiviert
- **Debug**: Logs in `offenheitscrawler.log` prüfen
- **Alternative**: Andere Suchbegriffe verwenden

**Problem: Crawling-Fehler**
```python
# In den Logs nach folgenden Meldungen suchen:
# "Direct connection successful" - OK
# "Using proxy fallback" - Proxy wird verwendet
# "All proxies failed" - Alle Proxies blockiert
```

**Problem: Zu wenige URLs gecrawlt**
- **Ursache**: Manche Webseiten blockieren automatisierte Zugriffe
- **Lösung**: Rate Limiting ist bereits implementiert
- **Workaround**: Einzelne Organisationen testen

### 🤖 Gemini CLI Modus

**Problem: Gemini Timeout**
```bash
# Timeout erhöhen (Standard: 120s)
gemini -p "Test" --timeout 180

# Netzwerkverbindung prüfen
ping google.com
```

**Problem: Rate Limit überschritten**
- **Gemini**: 10 Anfragen/Minute (automatisch begrenzt)
- **Lösung**: Warten oder kleinere Batches verwenden
- **Einstellung**: Rate Limiting auf 10+ Sekunden erhöhen

### 📊 Datenverarbeitung

**Problem: Encoding-Fehler**
```bash
# Windows: UTF-8 Konsole aktivieren
chcp 65001

# CSV-Dateien in Excel öffnen:
# 1. "Daten" > "Aus Text/CSV"
# 2. Datei auswählen
# 3. Encoding: "UTF-8" wählen
# 4. Trennzeichen: "Semikolon" wählen
```

**Problem: Speicher-Probleme**
- **Symptom**: "MemoryError" oder sehr langsame Verarbeitung
- **Lösung**: 
  - Kleinere Organisationslisten (max. 50 auf einmal)
  - Einzelne Organisationen zur Diagnose
  - Streamlit neu starten: `Ctrl+C` dann `streamlit run app.py`

**Problem: Visualisierungen werden nicht erstellt**
```python
# Plotly Installation prüfen
pip install plotly --upgrade

# Kaleido für PNG-Export
pip install kaleido

# Alternative: Nur interaktive Charts verwenden
```

### 📁 Datei-Probleme

**Problem: CSV-Dateien können nicht geladen werden**
- **Format prüfen**: `Name;URL` (Semikolon-getrennt)
- **Encoding**: UTF-8 ohne BOM
- **Beispiel-Zeile**: `Universität München;https://www.uni-muenchen.de`

**Problem: YAML-Kriterien werden nicht erkannt**
```yaml
# Korrekte YAML-Struktur prüfen:
kriterien:
  - name: "Kriterium 1"
    beschreibung: "Beschreibung"
    kategorie: "Kategorie"
    gewicht: 1.0
```

**Problem: Ergebnisse werden nicht gespeichert**
- **Berechtigungen**: Schreibrechte für `/results` Verzeichnis prüfen
- **Speicherplatz**: Ausreichend freier Speicherplatz
- **Pfad**: Keine Sonderzeichen in Verzeichnisnamen

### 🔍 Debug-Tipps

**Logging aktivieren:**
```python
# Detaillierte Logs anzeigen
tail -f offenheitscrawler.log  # Linux/macOS
Get-Content offenheitscrawler.log -Wait  # Windows PowerShell
```

**Schritt-für-Schritt Debugging:**
1. **Einzelne Organisation testen**: Nur 1 Organisation auswählen
2. **Einfache Kriterien**: Wenige Kriterien (3-5) verwenden
3. **Logs prüfen**: Fehlerdetails in `offenheitscrawler.log`
4. **API-Limits**: Rate Limiting auf 10+ Sekunden setzen
5. **Netzwerk**: Internetverbindung und Proxy-Einstellungen prüfen

**Häufige Fehlermeldungen:**
- `ModuleNotFoundError`: `pip install -r requirements.txt`
- `FileNotFoundError`: Dateipfade und Berechtigungen prüfen
- `JSONDecodeError`: API-Antwort ungültig, Rate Limiting erhöhen
- `ConnectionError`: Netzwerkprobleme, Proxy-Einstellungen prüfen
- `TimeoutError`: Timeout-Werte erhöhen

## 📝 Logging

Alle Aktivitäten werden detailliert in `offenheitscrawler.log` protokolliert:
- API-Aufrufe und Antwortzeiten
- Fehler und Warnungen
- Verarbeitungsfortschritt
- Datei-Speicheroperationen

## 🤝 Beitragen

Beiträge sind willkommen! Bitte:
1. Fork des Repositories erstellen
2. Feature-Branch erstellen
3. Änderungen committen
4. Pull Request erstellen

## 📄 Lizenz

Dieses Projekt steht unter der Apache 2.0 Lizenz.

## 🆘 Support

Bei Problemen oder Fragen:
1. Logs in `offenheitscrawler.log` prüfen
2. GitHub Issues erstellen
3. Dokumentation nochmals durchlesen

---

**Entwickelt mit ❤️ für transparente Organisationsanalyse**
