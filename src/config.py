"""Gemeinsame Einstellungen fuer alle Skripte (Pfade und die 52 Klassen).

Klassennamen laut Drehbuch: farbe_wert, klein, ohne Umlaute.
Die Reihenfolge der IDs 0 ... 51 wird direkt aus Label Studios classes.txt
uebernommen, damit die Label-Dateien und data.yaml exakt uebereinstimmen.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "raw_data"                 # Originalfotos (beliebige Unterordner)
LABELS_RAW = RAW / "labels_raw"         # YOLO-Export(e) aus Label Studio
SERIEN_CSV = RAW / "serien.csv"         # Zuordnung Foto -> Aufnahmeserie
DATASET = ROOT / "dataset"              # fertiger YOLO-Datensatz
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PARTS = ("train", "val", "test")

FARBEN = ["kreuz", "pik", "herz", "karo"]
WERTE = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "bube", "dame", "koenig", "ass"]

# Klassenreihenfolge direkt aus Label Studios classes.txt laden
CLASSES_FILE = RAW / "classes.txt"
if not CLASSES_FILE.exists():
    CLASSES_FILE = LABELS_RAW / "classes.txt"

if CLASSES_FILE.exists():
    CLASSES = [
        z.strip()
        for z in CLASSES_FILE.read_text(encoding="utf-8").splitlines()
        if z.strip()
    ]
else:
    # Fallback auf Drehbuch-Reihenfolge, falls noch keine classes.txt existiert
    CLASSES = [f"{f}_{w}" for f in FARBEN for w in WERTE]

assert len(CLASSES) == 52, f"Erwartet 52 Klassen, gefunden: {len(CLASSES)}"


def list_images(folder: Path, recursive: bool = False):
    """Alle Bilddateien eines Ordners, sortiert."""
    it = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMG_EXT)


def clear_folder(folder: Path):
    """Loescht alle Dateien im Ordner, laesst aber .gitkeep stehen."""
    folder.mkdir(parents=True, exist_ok=True)
    for f in folder.iterdir():
        if f.is_file() and f.name != ".gitkeep":
            f.unlink()