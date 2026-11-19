# Poker-KI
Entwicklung einer lokalen KI-Anwendung zur Echtzeit-Erkennung von Pokerkarten

## Projektstruktur

- `raw_data/`: unbearbeitete Fotos und exportierte Label-Studio-Labels
- `dataset/`: fertiger YOLO-Datensatz mit Train-, Validierungs- und Testdaten
- `models/`: lokale Modellgewichte
- `src/`: Datenvorbereitung, Training, Evaluation und Anwendung
- `results/`: automatisch erzeugte Auswertungen

## Einrichtung

### 1. Python-Umgebung erstellen

Voraussetzung ist Python 3.10 oder neuer. Im Projektordner:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Rohdaten ablegen

Fotos nach Aufnahmeserie in `raw_data/serie_01/` und `raw_data/serie_02/`
ablegen. Die exportierten YOLO-Labels aus Label Studio gehören nach
`raw_data/labels_raw/`.

### 3. Datensatz vorbereiten

Die Skripte in `src/data_prep/` ausführen. Dabei müssen die Pfade und
Einstellungen in den Skripten an die eigenen Fotos angepasst werden:

```powershell
python src/data_prep/split_dataset.py
python src/data_prep/augment.py
python src/data_prep/count_classes.py
```

Augmentation wird ausschließlich auf den Trainingsdaten durchgeführt.
Validierungs- und Testbilder bleiben unbearbeitet.

### 4. Modell trainieren

Das vortrainierte YOLO-Modell `yolov8n.pt` nach `models/` legen und anschließend
trainieren:

```powershell
python src/training/train.py
python src/training/evaluate.py
```

Die Evaluation schreibt Kennzahlen und Grafiken nach `results/`.

### 5. Poker-Assistent starten

Nach erfolgreichem Training muss das erzeugte `best.pt` in `models/` liegen:

```powershell
python src/app/main.py
```

Die Anwendung verwendet die lokale Kamera und verarbeitet alle Daten lokal.

## Offline-Einrichtung

Für eine vollständig offlinefähige Einrichtung müssen die in
`requirements.txt` genannten Wheels sowie `yolov8n.pt` vorab heruntergeladen
und lokal bereitgestellt werden. Danach kann die Installation beispielsweise
mit einem lokalen Wheel-Verzeichnis erfolgen:

```powershell
pip install --no-index --find-links .\wheels -r requirements.txt
```
