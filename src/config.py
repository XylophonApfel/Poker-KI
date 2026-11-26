"""Gemeinsame Einstellungen fuer alle Skripte (Pfade und die 52 Klassen).

Klassennamen laut Drehbuch: farbe_wert, klein, ohne Umlaute.
Feste Reihenfolge: Kreuz, Pik, Herz, Karo, jeweils 2 ... Ass  ->  IDs 0 ... 51.
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
CLASSES = [f"{f}_{w}" for f in FARBEN for w in WERTE]   # 0: kreuz_2 ... 51: karo_ass
assert len(CLASSES) == 52


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
