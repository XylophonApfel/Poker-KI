"""Erstellt raw_data/serien.csv: welches Foto gehoert zu welcher Aufnahmeserie.

Warum? Der Split (70/10/20) muss ganze Serien zusammenhalten (Drehbuch Abschnitt 5),
sonst landen fast gleiche Fotos in train UND test und die Ergebnisse sind geschoent.

Zwei Wege:
  a) Fotos liegen schon in Serienordnern (raw_data/serie_01/, raw_data/serie_02/, ...)
     -> Serie = Ordnername.
  b) Alle Fotos liegen in einem Ordner (z. B. raw_data/fotos/)
     -> Serien werden ueber die Aufnahmezeit erkannt: Liegt zwischen zwei Fotos eine
        Pause von mehr als --pause Minuten, beginnt eine neue Serie.
        (Aufnahmezeit aus den EXIF-Daten, sonst Datei-Aenderungsdatum.)

Die CSV danach in Excel/VS Code kontrollieren und bei Bedarf von Hand korrigieren.
Die Spalten deck/hintergrund/licht/notiz koennt ihr als Aufnahmeprotokoll ausfuellen.

Aufruf:
  python src/data_prep/serien_erstellen.py                 # automatisch
  python src/data_prep/serien_erstellen.py --pause 3       # andere Pausenlaenge
  python src/data_prep/serien_erstellen.py --ueberschreiben
"""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import LABELS_RAW, RAW, SERIEN_CSV, list_images  # noqa: E402

from PIL import Image  # noqa: E402

SPALTEN = ["datei", "serie", "deck", "hintergrund", "licht", "notiz"]


def aufnahmezeit(path: Path) -> datetime:
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            raw = exif.get_ifd(0x8769).get(36867) or exif.get(306)   # DateTimeOriginal / DateTime
            if raw:
                return datetime.strptime(str(raw).strip("\x00 "), "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pause", type=float, default=2.0, help="Minuten Pause, ab der eine neue Serie beginnt")
    ap.add_argument("--ueberschreiben", action="store_true", help="vorhandene serien.csv ersetzen")
    args = ap.parse_args()

    if SERIEN_CSV.exists() and not args.ueberschreiben:
        raise SystemExit(f"{SERIEN_CSV} gibt es schon. Mit --ueberschreiben neu erzeugen "
                         "(eure Handkorrekturen gehen dann verloren).")

    images = [p for p in list_images(RAW, recursive=True) if LABELS_RAW not in p.parents]
    if not images:
        raise SystemExit(f"Keine Fotos in {RAW} gefunden.")

    rows = []
    # a) Fotos in Unterordnern serie_XX -> Ordnername
    in_series_dir = [p for p in images if p.parent != RAW and p.parent.name.lower().startswith("serie")]
    for p in in_series_dir:
        rows.append({"datei": p.name, "serie": p.parent.name})

    # b) restliche Fotos -> nach Aufnahmezeit gruppieren
    rest = sorted((aufnahmezeit(p), p) for p in images if p not in set(in_series_dir))
    nr, last = 0, None
    for t, p in rest:
        if last is None or (t - last).total_seconds() > args.pause * 60:
            nr += 1
        last = t
        rows.append({"datei": p.name, "serie": f"zeit_{nr:02d}"})

    names = [r["datei"] for r in rows]
    doppelt = sorted({n for n in names if names.count(n) > 1})
    if doppelt:
        raise SystemExit("Dateinamen muessen eindeutig sein, doppelt sind: " + ", ".join(doppelt[:10]))

    with open(SERIEN_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=SPALTEN, delimiter=";")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    from collections import Counter
    c = Counter(r["serie"] for r in rows)
    print(f"{len(rows)} Fotos in {len(c)} Serien -> {SERIEN_CSV}")
    for s, n in sorted(c.items()):
        print(f"  {s}: {n} Fotos")
    if rest and len(c) < 10:
        print("Hinweis: Sehr wenige Serien. Fuer einen sauberen Split braucht ihr mind. ~10, besser 30+.\n"
              "  -> --pause kleiner waehlen oder serien.csv von Hand anpassen.")


if __name__ == "__main__":
    main()
