"""Teilt die gelabelten Originalfotos nach Aufnahmeserien in train/val/test (70/10/20).

Erwartet:
  raw_data/**/<foto>.jpg          Originalfotos, beliebig in Unterordnern (Dateinamen eindeutig!)
  raw_data/serien.csv             Foto -> Serie (siehe serien_erstellen.py).
                                  Fehlt die Datei, gilt der Ordnername als Serie.
  raw_data/labels_raw/**/*.txt    YOLO-Export(e) aus Label Studio, jeweils mit classes.txt
                                  (z. B. labels_raw/pavi/ und labels_raw/aron/ mit je eigener classes.txt)

Erzeugt:
  dataset/images/{train,val,test}, dataset/labels/{train,val,test}
  dataset/data.yaml, dataset/split_info.csv

Regeln (Drehbuch Abschnitt 5):
  - Eine Serie landet immer komplett in genau einem Teil.
  - Der Split passiert VOR der Augmentation.
  - Unter allen zufaelligen Aufteilungen wird die genommen, bei der moeglichst jede
    Klasse in val und test vorkommt.

Ausserdem:
  - Label Studio setzt vor Dateinamen oft '1a2b3c4d-'. Das wird entfernt.
  - Label Studio schreibt classes.txt oft in anderer Reihenfolge. Die IDs werden auf
    unsere feste Reihenfolge 0 = kreuz_2 ... 51 = karo_ass umgeschrieben.
  - Fotos werden auf hoechstens MAX_SIDE Pixel verkleinert (YOLO trainiert ohnehin mit 640).
    Das spart Platz und macht Augmentation/Training auf dem Laptop viel schneller.
    Dabei wird die Handy-Drehung (EXIF) fest ins Bild uebernommen, genau wie Label Studio
    das Foto beim Labeln angezeigt hat.
"""
import argparse
import csv
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASSES, DATASET, LABELS_RAW, PARTS, RAW, SERIEN_CSV, clear_folder, list_images  # noqa: E402

RATIOS = {"train": 0.7, "val": 0.1, "test": 0.2}
MIN_VAL, MIN_TEST = 1, 3       # gewuenschte Mindestanzahl Karten pro Klasse in val / test
TRIES = 3000                   # Anzahl zufaelliger Aufteilungen, die beste wird genommen
SEED = 42
MAX_SIDE = 1280                # 0 = Originalgroesse behalten
JPG_QUALITY = 92


def strip_ls_prefix(stem: str) -> str:
    """Label Studio setzt beim Upload oft '1a2b3c4d-' vor den Dateinamen."""
    return re.sub(r"^[0-9a-f]{8}-", "", stem)


def load_labels():
    """Liest alle Label-Dateien unter labels_raw/ (auch Unterordner) mit der jeweils
    naechstgelegenen classes.txt und schreibt die IDs auf unsere Reihenfolge um."""
    if not LABELS_RAW.exists():
        raise SystemExit(f"Ordner {LABELS_RAW} fehlt.")
    remaps = {}
    for cf in LABELS_RAW.rglob("classes.txt"):
        names = [l.strip() for l in cf.read_text(encoding="utf-8").splitlines() if l.strip()]
        unknown = [n for n in names if n not in CLASSES]
        if unknown:
            raise SystemExit(f"Unbekannte Klassennamen in {cf}: {unknown}\n"
                             "Erlaubt sind nur farbe_wert, z. B. herz_ass, pik_10, karo_koenig "
                             "(Label-Studio-Konfiguration: labelstudio_config.xml).")
        remaps[cf.parent] = {i: CLASSES.index(n) for i, n in enumerate(names)}
    if not remaps:
        raise SystemExit(f"Keine classes.txt in {LABELS_RAW} gefunden. Bitte aus dem Label-Studio-Export mitkopieren.")

    def remap_for(folder: Path):
        for d in [folder, *folder.parents]:
            if d in remaps:
                return remaps[d]
            if d == LABELS_RAW:
                break
        return None

    labels, problems = {}, []
    for txt in sorted(LABELS_RAW.rglob("*.txt")):
        if txt.name in ("classes.txt", "notes.txt"):
            continue
        remap = remap_for(txt.parent)
        if remap is None:
            problems.append(f"{txt}: keine classes.txt im selben oder uebergeordneten Ordner")
            continue
        rows = []
        for n, line in enumerate(txt.read_text(encoding="utf-8").splitlines(), 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5 or int(parts[0]) not in remap:
                problems.append(f"{txt.name} Zeile {n}: '{line}' ist kein gueltiges YOLO-Label")
                continue
            x, y, w, h = (min(max(float(v), 0.0), 1.0) for v in parts[1:])
            if w <= 0 or h <= 0:
                continue
            rows.append((remap[int(parts[0])], x, y, w, h))
        key = strip_ls_prefix(txt.stem)
        if key in labels:
            problems.append(f"Label fuer '{key}' doppelt vorhanden ({txt}) - die spaetere Datei gewinnt")
        labels[key] = rows
    if problems:
        print(f"WARNUNG: {len(problems)} Probleme in den Labels:")
        for p in problems[:15]:
            print("  " + p)
    return labels


def load_series(images):
    """Foto -> Serie, aus serien.csv oder (falls nicht vorhanden) aus dem Ordnernamen."""
    mapping = {}
    if SERIEN_CSV.exists():
        with open(SERIEN_CSV, encoding="utf-8-sig", newline="") as f:
            sample = f.read(2048); f.seek(0)
            delim = ";" if sample.count(";") >= sample.count(",") else ","
            for row in csv.DictReader(f, delimiter=delim):
                datei, serie = (row.get("datei") or "").strip(), (row.get("serie") or "").strip()
                if datei and serie:
                    mapping[Path(datei).stem] = serie
        print(f"Serien aus {SERIEN_CSV.name}: {len(set(mapping.values()))}")
    result, unknown = {}, []
    for img in images:
        if img.stem in mapping:
            result[img] = mapping[img.stem]
        elif img.parent != RAW:
            result[img] = img.parent.name
        else:
            unknown.append(img.name)
    if unknown:
        raise SystemExit(f"{len(unknown)} Fotos haben keine Serie (z. B. {unknown[:3]}).\n"
                         "  -> python src/data_prep/serien_erstellen.py ausfuehren oder serien.csv ergaenzen.")
    return result


def copy_image(src: Path, dst: Path):
    img = cv2.imread(str(src))          # liest inkl. EXIF-Drehung
    if img is None:
        raise SystemExit(f"Bild kann nicht gelesen werden: {src}")
    h, w = img.shape[:2]
    if MAX_SIDE and max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst.with_suffix(".jpg")), img, [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])


def write_data_yaml():
    # Kein "path:" -> Ultralytics nimmt den Ordner, in dem data.yaml liegt.
    # So funktioniert die Datei auf jedem Rechner, egal wo das Repo liegt.
    lines = ["# Automatisch erzeugt von src/data_prep/split_dataset.py",
             "train: images/train", "val: images/val", "test: images/test", "",
             f"nc: {len(CLASSES)}", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(CLASSES)]
    (DATASET / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    global MAX_SIDE
    ap = argparse.ArgumentParser(description="Split nach Serien (70/10/20)")
    ap.add_argument("--max-side", type=int, default=MAX_SIDE, help="laengste Bildseite in Pixel (0 = Original)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    MAX_SIDE = args.max_side

    labels = load_labels()
    all_imgs = [p for p in list_images(RAW, recursive=True) if LABELS_RAW not in p.parents]
    stems = Counter(p.stem for p in all_imgs)
    dup = [s for s, n in stems.items() if n > 1]
    if dup:
        raise SystemExit(f"Dateinamen muessen eindeutig sein. Doppelt: {dup[:10]}")

    labelled = [p for p in all_imgs if p.stem in labels]
    missing = [p for p in all_imgs if p.stem not in labels]
    orphan = sorted(set(labels) - set(stems))
    if missing:
        print(f"WARNUNG: {len(missing)} Fotos ohne Label werden ignoriert, z. B. {[p.name for p in missing[:3]]}")
    if orphan:
        print(f"WARNUNG: {len(orphan)} Labels ohne passendes Foto, z. B. {orphan[:3]}")
    if not labelled:
        raise SystemExit("Keine gelabelten Fotos gefunden. Stimmen die Dateinamen von Fotos und Labels ueberein?")

    serie_of = load_series(labelled)
    series = defaultdict(list)
    for img in labelled:
        series[serie_of[img]].append(img)
    if len(series) < 3:
        raise SystemExit(f"Nur {len(series)} Serie(n) gefunden - fuer train/val/test braucht ihr mindestens 3, besser 30+.")

    sc = {s: Counter(r[0] for img in imgs for r in labels[img.stem]) for s, imgs in series.items()}
    n_img = {s: len(imgs) for s, imgs in series.items()}
    total = sum(n_img.values())

    # viele zufaellige Serien-Aufteilungen ausprobieren, die beste behalten
    rng = random.Random(args.seed)
    names = sorted(series)
    best = None
    for _ in range(TRIES):
        rng.shuffle(names)
        assign, filled = {}, {k: 0 for k in RATIOS}
        for s in names:
            # Serie in den Teil legen, der relativ am weitesten unter seinem Ziel liegt
            part = min(RATIOS, key=lambda k: filled[k] / (RATIOS[k] * total))
            assign[s] = part
            filled[part] += n_img[s]
        cnt = {k: Counter() for k in RATIOS}
        for s, part in assign.items():
            cnt[part].update(sc[s])
        penalty = sum(max(0, MIN_VAL - cnt["val"][c]) + max(0, MIN_TEST - cnt["test"][c]) for c in range(52))
        penalty += sum(abs(filled[k] / total - RATIOS[k]) for k in RATIOS) * 20
        if best is None or penalty < best[0]:
            best = (penalty, dict(assign), cnt, dict(filled))
    _, assign, cnt, filled = best

    # Ausgabe neu schreiben (.gitkeep bleibt stehen)
    for sub in ("images", "labels"):
        for part in PARTS:
            clear_folder(DATASET / sub / part)
    for i, (s, part) in enumerate(sorted(assign.items()), 1):
        print(f"\rKopiere Serie {i}/{len(assign)} ...", end="", flush=True)
        for img in series[s]:
            copy_image(img, DATASET / "images" / part / f"{img.stem}.jpg")
            lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in labels[img.stem]]
            (DATASET / "labels" / part / f"{img.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print()

    with open(DATASET / "split_info.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["serie", "teil", "fotos", "karten"])
        for s in sorted(assign):
            w.writerow([s, assign[s], n_img[s], sum(sc[s].values())])
    write_data_yaml()

    print(f"Fotos: {total} in {len(series)} Serien  ->  " +
          ", ".join(f"{k}: {filled[k]} ({filled[k] / total:.0%})" for k in RATIOS))
    weak = [CLASSES[c] for c in range(52) if cnt["val"][c] < MIN_VAL or cnt["test"][c] < MIN_TEST]
    if weak:
        print(f"HINWEIS: {len(weak)} Klassen haben wenig Karten in val/test: {', '.join(weak)}")
        print("  -> Diese Karten in einer neuen Serie nachfotografieren, dann Split erneut starten.")
    else:
        print("Alle 52 Klassen sind in val und test vertreten.")
    print(f"Aufteilung: {DATASET / 'split_info.csv'}")
    print("Naechster Schritt: python src/data_prep/augment.py")


if __name__ == "__main__":
    main()
