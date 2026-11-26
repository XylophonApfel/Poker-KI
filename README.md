# Poker-KI

Lokale KI-Anwendung zur Echtzeit-Erkennung von Pokerkarten (YOLOv8, 52 Klassen).
Alles läuft offline auf dem eigenen Rechner, eine Grafikkarte ist optional.

**Ablauf in einem Satz:** Fotos machen → in Serien ordnen → in Label Studio labeln →
nach Serien aufteilen (70/10/20) → nur Trainingsbilder augmentieren → zählen → trainieren → evaluieren.

```
raw_data/                 Originalfotos (NICHT im Git) + serien.csv + Label-Studio-Export (im Git)
  serie_01/ …             entweder Fotos nach Serien in Unterordnern …
  fotos/                  … oder alle Fotos in einem Ordner (Serien dann über serien.csv)
  serien.csv              Foto → Aufnahmeserie (+ Aufnahmeprotokoll)
  labels_raw/             YOLO-Export aus Label Studio (.txt + classes.txt)
dataset/                  wird automatisch erzeugt (images|labels/train|val|test + data.yaml)
models/                   yolov8n.pt (Startmodell), best.pt (unser trainiertes Modell)
results/                  Zähltabelle, Trainingsverlauf, Evaluation
src/config.py             Klassenliste und Pfade (für alle Skripte gleich)
src/data_prep/            serien_erstellen.py, split_dataset.py, augment.py, count_classes.py
src/training/             train.py, evaluate.py
src/app/                  Poker-Assistent (Kamera-App)
labelstudio_config.xml    Label-Studio-Vorlage mit genau unseren 52 Klassen
```

## Die 52 Klassen

Name = `farbe_wert`, klein, ohne Umlaute. Die Reihenfolge (= Klassen-ID) ist fest:

| IDs | Farbe | Werte |
|---|---|---|
| 0–12 | `kreuz` | `2 3 4 5 6 7 8 9 10 bube dame koenig ass` |
| 13–25 | `pik` | gleiche Reihenfolge |
| 26–38 | `herz` | gleiche Reihenfolge |
| 39–51 | `karo` | gleiche Reihenfolge |

Beispiele: `0 = kreuz_2`, `12 = kreuz_ass`, `37 = herz_koenig`, `51 = karo_ass`.
Die Liste steht in `src/config.py` und in `labelstudio_config.xml`.

---

## 1. Einmalig: Umgebung einrichten

Voraussetzung: Python 3.10 bis 3.13 (in VS Code mit Python-Erweiterung). PowerShell im Repo-Ordner öffnen:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

In VS Code unten rechts den Interpreter `.venv` auswählen (Strg+Shift+P → „Python: Select Interpreter“).

> Falls `Activate.ps1` wegen „Ausführung von Skripts deaktiviert“ abbricht:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` und dann nochmal.

**Nur mit NVIDIA-Grafikkarte:** *vor* `pip install -r requirements.txt` PyTorch mit CUDA installieren
(Befehl auf pytorch.org → „Get Started“ kopieren). Prüfen:
`python -c "import torch; print(torch.cuda.is_available())"` → `True`. Ohne GPU läuft alles auf der CPU, nur langsamer.

**Startmodell laden (einmal mit Internet, ca. 6 MB):**

```powershell
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
move yolov8n.pt models\
```

Danach läuft alles offline. Tipp: den ersten kurzen Probelauf (Schritt 6, `--epochs 1`) mit Internet machen,
dann lädt Ultralytics einmal eine Schriftart für die Grafiken nach.

Kontrolle: `python -c "import ultralytics, albumentations, sklearn, cv2; print('ok')"` → `ok`

**Bekannte Stolperfalle:** Albumentations bringt `opencv-python-headless` mit. Falls die App später bei
`cv2.imshow` mit „The function is not implemented“ abbricht:

```powershell
pip uninstall -y opencv-python-headless opencv-python
pip install opencv-python
```

---

## 2. Fotos ablegen und Serien festlegen

Eine **Serie** = mehrere Fotos derselben Szene (gleiche Karten, gleicher Hintergrund, gleiches Licht,
nur Winkel/Abstand anders). Damit die Testergebnisse ehrlich sind, landet eine Serie **immer komplett**
in train, val oder test.

Fotos **nicht ins Git** legen (zu groß). Zwischen euch per USB-Stick oder Cloud-Ordner austauschen.
Dateinamen müssen über alle Fotos **eindeutig** sein (Handy-Namen wie `IMG_1234.jpg` sind meist schon eindeutig,
solange ihr nicht zwei Handys mit gleichen Nummern mischt: dann vorher umbenennen, z. B. `pavi_IMG_1234.jpg`).

Zwei Möglichkeiten, wählt eine:

**Variante A: Serienordner (am einfachsten)**

```
raw_data/serie_01/IMG_1234.jpg …
raw_data/serie_02/IMG_1250.jpg …
```

Fertig, die Serie ist der Ordnername. `serien.csv` braucht ihr dann nicht (könnt sie aber trotzdem erzeugen).

**Variante B: alle Fotos in einen Ordner + `serien.csv`**

```
raw_data/fotos/IMG_1234.jpg …
```

```powershell
python src/data_prep/serien_erstellen.py
```

Das Skript erkennt Serien über die Aufnahmezeit: liegt zwischen zwei Fotos mehr als 2 Minuten Pause,
beginnt eine neue Serie (`--pause 3` für 3 Minuten). Danach `raw_data/serien.csv` in Excel öffnen und
kontrollieren. Die Spalte `serie` könnt ihr frei ändern, die Spalten `deck`, `hintergrund`, `licht`, `notiz`
sind euer **Aufnahmeprotokoll** für die Doku.

Ziel laut Drehbuch: 350–400 Fotos, ca. 30–40 Serien, jede Karte mindestens 40-mal sichtbar.

---

## 3. Labeln mit Label Studio

Label Studio in einer **eigenen** Umgebung installieren (verträgt sich nicht mit den anderen Paketen):

```powershell
python -m venv .venv-ls
.\.venv-ls\Scripts\Activate.ps1
pip install label-studio
label-studio start
```

1. Browser öffnet http://localhost:8080 → Konto anlegen → Projekt „Poker-KI“.
2. *Settings → Labeling Interface → Code*: Inhalt von `labelstudio_config.xml` einfügen → Speichern.
3. *Import*: Fotos hochladen, **höchstens ca. 100 auf einmal** (größere Mengen bricht Label Studio ab).
   Das läuft lokal, braucht also kein Internet.
4. **Labelregel (Drehbuch):** Karte nur markieren, wenn mindestens eine Ecke mit Wert **und** Farbsymbol
   vollständig sichtbar ist. Box um den sichtbaren Teil der Karte. Rückseiten nicht markieren.
5. *Export → YOLO*. Aus der ZIP-Datei den Ordner `labels/` **und** `classes.txt` nach
   `raw_data/labels_raw/<euer_name>/` kopieren, z. B.:

```
raw_data/labels_raw/pavi/classes.txt
raw_data/labels_raw/pavi/1a2b3c4d-IMG_1234.txt …
raw_data/labels_raw/aron/classes.txt
raw_data/labels_raw/aron/…
```

Die Bilder aus dem Export braucht ihr nicht. Arbeitet ihr zu zweit, labelt jeder seine eigenen Serien in seinem
eigenen Label-Studio-Projekt (gleiche Konfiguration) und legt seinen Export in einen eigenen Unterordner.

Das Split-Skript erledigt automatisch:
- das Kürzel `1a2b3c4d-`, das Label Studio vor die Dateinamen setzt, wird entfernt;
- die Klassen-IDs werden von der Reihenfolge in `classes.txt` auf unsere feste Reihenfolge 0–51 umgeschrieben
  (jeder Unterordner mit seiner eigenen `classes.txt`).

**Labels gehören ins Git** (nur kleine Textdateien). So hat jeder die Labels des anderen ohne große Uploads.

---

## 4. Datensatz aufteilen (70/10/20 nach Serien)

```powershell
python src/data_prep/split_dataset.py
```

- probiert 3000 zufällige Aufteilungen der Serien und nimmt die, bei der möglichst jede Karte in val und test vorkommt;
- verkleinert die Fotos auf max. 1280 px (YOLO trainiert ohnehin mit 640 px, spart viel Zeit und Platz;
  Originalgröße mit `--max-side 0`);
- übernimmt die Handy-Drehung fest ins Bild (so, wie Label Studio das Foto beim Labeln gezeigt hat);
- schreibt `dataset/images|labels/train|val|test`, `dataset/data.yaml` und `dataset/split_info.csv` (welche Serie wo).

Meldet das Skript Fotos ohne Label oder Klassen mit zu wenig Karten in val/test, nachlabeln bzw. nachfotografieren
und das Skript einfach erneut starten. Es baut `dataset/` jedes Mal komplett neu.

## 5. Augmentation (nur train) und Zähltabelle

```powershell
python src/data_prep/augment.py          # 4 Varianten pro Trainingsbild (Drehbuch: 3–5, z. B. --n 3)
python src/data_prep/count_classes.py
```

Augmentation laut Drehbuch: Drehen (90°-Schritte und ±30°), Zuschneiden, Skalieren, Helligkeit/Kontrast,
Rauschen, minimale Farbtonverschiebung. **Kein Spiegeln.** val und test bleiben unverändert.
Bitte Stichprobe in `results/augment_preview/` ansehen (Boxen müssen auf den Karten sitzen).

`count_classes.py` schreibt `results/zaehltabelle.csv` (pro Klasse: train Original, train augmentiert, val, test,
gesamt) und das Diagramm `results/zaehltabelle.png`. Beides kommt in die Doku.

Wichtig: Nach jedem neuen `split_dataset.py` auch `augment.py` und `count_classes.py` neu ausführen.

## 6. Training

Erst ein kurzer Probelauf, um die Zeit pro Epoche zu messen:

```powershell
python src/training/train.py --epochs 1
```

Dann das richtige Training (am besten über Nacht, Laptop ans Netzteil, Energiesparen aus):

```powershell
python src/training/train.py
```

- Standard: 80 Epochen, bricht automatisch ab, wenn sich 20 Epochen nichts verbessert.
- Erkennt automatisch, ob eine NVIDIA-GPU da ist (sonst CPU, Batch 8).
- Grobe Dauer auf einem normalen CPU-Laptop: ca. 3–8 Minuten pro Epoche bei ~1.500 Trainingsbildern.
  Zu langsam? `--epochs 40` oder `--imgsz 512`.
- Abgebrochen (Absturz, Laptop zugeklappt)? Weiter mit `python src/training/train.py --resume`.
- Spiegeln und Farbtonverschiebung sind auch in der YOLO-eigenen Augmentation ausgeschaltet.

Ergebnis: `models/best.pt` und `results/train/` (Verlauf, Grafiken von Ultralytics).

## 7. Evaluation

```powershell
python src/training/evaluate.py
```

Alles landet in `results/evaluation/`:

| Datei | Inhalt |
|---|---|
| `classification_report.txt` | Accuracy gesamt, Precision/Recall/F1 pro Klasse und Durchschnitt, mAP50 auf test |
| `kennzahlen_pro_klasse.csv` | Accuracy, Precision, Recall, F1 pro Klasse (Excel-tauglich) |
| `konfusionsmatrix_52.png` | Konfusionsmatrix aller 52 Karten (+ „nicht_erkannt“, „fehlalarm“) |
| `konfusionsmatrix_farbe.png`, `konfusionsmatrix_wert.png` | verdichtet nach Farbe bzw. Wert |
| `trainingsverlauf.png` | Loss (Training/Validierung) und Accuracy/Precision/Recall/mAP50 pro Epoche |
| `verwechslungen.txt` | häufigste Verwechslungen, z. B. `pik_6 -> pik_9: 3x` |
| `yolo_test/` | Standard-Grafiken von Ultralytics (PR-Kurve, F1-Kurve, Beispielbilder) |

So wird gezählt: Jede erkannte Karte wird der markierten Karte mit der größten Überlappung zugeordnet
(IoU ≥ 0,5). Accuracy = richtig erkannte Karten / alle markierten Karten im Testset.
Die Sicherheitsschwelle ist dieselbe wie in der App (`conf = 0.6`), ändern mit `--conf 0.5`.

## 8. Poker-Assistent starten

```powershell
python src/app/main.py
```

Benötigt `models/best.pt` und eine Kamera.

---

## Kurzfassung: alle Befehle hintereinander

```powershell
.\.venv\Scripts\Activate.ps1
python src/data_prep/serien_erstellen.py     # nur bei Variante B (alle Fotos in einem Ordner)
python src/data_prep/split_dataset.py
python src/data_prep/augment.py
python src/data_prep/count_classes.py
python src/training/train.py
python src/training/evaluate.py
```

## Was ins Git kommt und was nicht

| ins Git | nicht ins Git (steht in `.gitignore`) |
|---|---|
| Code (`src/`), `README.md`, `requirements.txt` | Fotos (`raw_data/**/*.jpg` usw.) |
| `raw_data/serien.csv` | `dataset/images`, `dataset/labels` (wird neu erzeugt) |
| `raw_data/labels_raw/**` (Label-Studio-Export, nur Text) | `models/*.pt` (Modelle) |
| `results/zaehltabelle.*`, `results/evaluation/*` (für die Doku) | `results/train/`, `results/augment_preview/` |

`best.pt` (ca. 6 MB) gebt ihr euch per USB-Stick weiter.

## Offline-Installation auf einem anderen Rechner

Auf einem Rechner mit Internet die Pakete herunterladen und per USB-Stick mitnehmen:

```powershell
pip download -r requirements.txt -d wheels
```

Auf dem Zielrechner (gleiche Python-Version):

```powershell
pip install --no-index --find-links .\wheels -r requirements.txt
```
