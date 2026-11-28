"""Erzeugt synthetische Multi-Karten-Trainingsbilder inkl. automatischer YOLO-Labels.

Schneidet Karten ausschliesslich aus den echten Trainingsbildern (dataset/images/train) aus
und platziert sie neu skaliert und gedreht auf zufaelligen Hintergruenden.
Optional: Leere Tischfotos (ohne Karten, ohne Labels!) in raw_data/backgrounds/ ablegen.
"""
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATASET, RAW, list_images  # noqa: E402

ANZAHL_BILDER = 600          # 600 Bilder x ~5-6 Karten = ~3.300 neue Kartenbeispiele (~60 pro Klasse)
KARTEN_PRO_BILD = (4, 7)     # Min. und Max. Karten pro erzeugtem Bild
CANVAS_GROESSE = 800         # Passend zu imgsz=800
BG_DIR = RAW / "backgrounds" # Optionaler Ordner fuer echte leere Tischfotos

random.seed(42)
np.random.seed(42)


def bild_laden(pfad: Path):
    daten = np.fromfile(str(pfad), dtype=np.uint8)
    return cv2.imdecode(daten, cv2.IMREAD_COLOR)


def bild_speichern(pfad: Path, bild):
    ok, puffer = cv2.imencode(".jpg", bild, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if ok:
        puffer.tofile(str(pfad))


def zufalls_hintergrund(echte_hintergruende, groesse=CANVAS_GROESSE):
    if echte_hintergruende and random.random() < 0.7:
        bg = random.choice(echte_hintergruende)
        return cv2.resize(bg, (groesse, groesse))

    # Realistische Tischfarben (Gruen, Holzbraun, Dunkelgrau, Hellgrau, Weiss, Schwarz)
    farben = [
        (35, 95, 45), (25, 70, 30), (50, 75, 115),
        (45, 45, 48), (110, 110, 115), (215, 215, 215), (20, 20, 22),
    ]
    basis = np.array(random.choice(farben), dtype=np.float32)
    jitter = np.random.uniform(-20, 20, size=3)
    farbe = np.clip(basis + jitter, 0, 255).astype(np.uint8)
    canvas = np.full((groesse, groesse, 3), farbe, dtype=np.uint8)

    # Leichtes Rauschen / Beleuchtungsverlauf hinzufuegen
    rauschen = np.random.normal(0, 8, canvas.shape).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + rauschen, 0, 255).astype(np.uint8)
    return canvas


def karten_aus_train_ausschneiden():
    train_img_dir = DATASET / "images" / "train"
    train_lbl_dir = DATASET / "labels" / "train"
    karten_pro_klasse = defaultdict(list)

    for img_pfad in list_images(train_img_dir):
        # Keine bereits augmentierten oder synthetischen Bilder verwenden
        if img_pfad.stem.startswith(("aug_", "syn_")):
            continue
        lbl_pfad = train_lbl_dir / f"{img_pfad.stem}.txt"
        if not lbl_pfad.exists():
            continue

        bild = bild_laden(img_pfad)
        if bild is None:
            continue
        h, w = bild.shape[:2]

        for zeile in lbl_pfad.read_text(encoding="utf-8").splitlines():
            teile = zeile.strip().split()
            if len(teile) != 5:
                continue
            cid = int(float(teile[0]))
            xc, yc, bw, bh = [float(x) for x in teile[1:]]
            x1 = max(0, int((xc - bw / 2) * w))
            y1 = max(0, int((yc - bh / 2) * h))
            x2 = min(w, int((xc + bw / 2) * w))
            y2 = min(h, int((yc + bh / 2) * h))
            if (x2 - x1) < 25 or (y2 - y1) < 25:
                continue
            karten_pro_klasse[cid].append(bild[y1:y2, x1:x2].copy())

    return karten_pro_klasse


def karte_transformieren(crop):
    ch, cw = crop.shape[:2]
    # Einheitliche, realistische Tischgroesse (150 bis 280 Pixel Hoehe auf 800x800)
    ziel_h = random.randint(150, 280)
    ziel_w = max(20, int(cw * (ziel_h / ch)))
    crop = cv2.resize(crop, (ziel_w, ziel_h), interpolation=cv2.INTER_AREA)

    # Leichte Helligkeits- und Kontrastvariation
    alpha = random.uniform(0.80, 1.20)
    beta = random.uniform(-20, 20)
    crop = np.clip(crop.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # Leichte Drehung ohne schwarze Ecken (ueber Maske)
    winkel = random.uniform(-25, 25)
    maske = np.full((ziel_h, ziel_w), 255, dtype=np.uint8)
    M = cv2.getRotationMatrix2D((ziel_w / 2, ziel_h / 2), winkel, 1.0)
    cos_v, sin_v = abs(M[0, 0]), abs(M[0, 1])
    nw = int(ziel_h * sin_v + ziel_w * cos_v)
    nh = int(ziel_h * cos_v + ziel_w * sin_v)
    M[0, 2] += (nw / 2) - ziel_w / 2
    M[1, 2] += (nh / 2) - ziel_h / 2

    rot_crop = cv2.warpAffine(crop, M, (nw, nh))
    rot_maske = cv2.warpAffine(maske, M, (nw, nh))
    return rot_crop, rot_maske


def ueberlappt_zu_stark(box, bestehende_boxen, max_iou=0.12):
    x1, y1, x2, y2 = box
    flaeche_a = (x2 - x1) * (y2 - y1)
    for bx1, by1, bx2, by2 in bestehende_boxen:
        ix1, iy1 = max(x1, bx1), max(y1, by1)
        ix2, iy2 = min(x2, bx2), min(y2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter > 0:
            flaeche_b = (bx2 - bx1) * (by2 - by1)
            iou = inter / (flaeche_a + flaeche_b - inter)
            if iou > max_iou:
                return True
    return False


def main():
    karten_pro_klasse = karten_aus_train_ausschneiden()
    klassen_ids = sorted(karten_pro_klasse.keys())
    if not klassen_ids:
        raise SystemExit("Keine Trainingskarten gefunden. Erst split_dataset.py ausfuehren.")

    print(f"Ausgeschnittene Kartenklassen aus train: {len(klassen_ids)}/52")

    # Optionale echte Tisch-Hintergruende laden
    echte_bgs = []
    if BG_DIR.exists():
        for p in list_images(BG_DIR):
            b = bild_laden(p)
            if b is not None:
                echte_bgs.append(b)
        print(f"Optionale echte Hintergruende geladen: {len(echte_bgs)}")

    train_img_dir = DATASET / "images" / "train"
    train_lbl_dir = DATASET / "labels" / "train"

    # Alte synthetische Dateien entfernen
    for alt in train_img_dir.glob("syn_*.jpg"):
        alt.unlink()
    for alt in train_lbl_dir.glob("syn_*.txt"):
        alt.unlink()

    erzeugt = 0
    for i in range(ANZAHL_BILDER):
        canvas = zufalls_hintergrund(echte_bgs)
        anzahl = random.randint(*KARTEN_PRO_BILD)
        platzierte_boxen = []
        yolo_zeilen = []

        for _ in range(anzahl):
            cid = random.choice(klassen_ids)
            crop = random.choice(karten_pro_klasse[cid])
            rot_crop, rot_maske = karte_transformieren(crop)
            nh, nw = rot_crop.shape[:2]
            if nw >= CANVAS_GROESSE - 20 or nh >= CANVAS_GROESSE - 20:
                continue

            # Freie Position auf dem Tisch suchen
            for _versuch in range(25):
                x1 = random.randint(10, CANVAS_GROESSE - nw - 10)
                y1 = random.randint(10, CANVAS_GROESSE - nh - 10)
                box = (x1, y1, x1 + nw, y1 + nh)
                if not ueberlappt_zu_stark(box, platzierte_boxen):
                    roi = canvas[y1:y1 + nh, x1:x1 + nw]
                    maske_bool = rot_maske > 127
                    roi[maske_bool] = rot_crop[maske_bool]
                    canvas[y1:y1 + nh, x1:x1 + nw] = roi
                    platzierte_boxen.append(box)

                    xc = (x1 + nw / 2) / CANVAS_GROESSE
                    yc = (y1 + nh / 2) / CANVAS_GROESSE
                    bw = nw / CANVAS_GROESSE
                    bh = nh / CANVAS_GROESSE
                    yolo_zeilen.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                    break

        if yolo_zeilen:
            basis = f"syn_{i:04d}"
            bild_speichern(train_img_dir / f"{basis}.jpg", canvas)
            (train_lbl_dir / f"{basis}.txt").write_text("\n".join(yolo_zeilen), encoding="utf-8")
            erzeugt += 1

    print(f"Fertig! {erzeugt} synthetische Trainingsbilder + Labels in dataset/images/train erzeugt.")


if __name__ == "__main__":
    main()