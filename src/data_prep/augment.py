"""Erzeugt aus jedem Trainings-Originalbild N Varianten (NUR train, nie val/test!).

Erlaubt (Lastenheft 3.3 / Drehbuch 4): Drehen, Zuschneiden/Skalieren, Helligkeit/Kontrast,
Rauschen, minimale Farbtonverschiebung. Verboten: Spiegeln (Karten waeren spiegelverkehrt),
starke Farbverschiebung (rot/schwarz muss erkennbar bleiben).
Bounding Boxes werden automatisch mittransformiert.
Zur Kontrolle landen einige Varianten mit eingezeichneten Boxen in results/augment_preview/.

Aufruf:
  python src/data_prep/augment.py              # 4 Varianten pro Bild
  python src/data_prep/augment.py --n 3
"""
import argparse
import random
import sys
from pathlib import Path

import albumentations as A
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASSES, DATASET, RESULTS, list_images  # noqa: E402

IMG_DIR = DATASET / "images" / "train"
LBL_DIR = DATASET / "labels" / "train"
PREVIEW = RESULTS / "augment_preview"

N_VARIANTS = 4           # Drehbuch: 3 bis 5 Varianten pro Original
MAX_ANGLE = 30           # freie Drehung +-30 Grad, dazu zufaellig 0/90/180/270 Grad (Boxen bleiben exakt)
N_PREVIEW = 20
SEED = 42


def build_transform(seed):
    return A.Compose(
        [
            A.RandomRotate90(p=0.5),                                   # Drehen um 90/180/270 Grad (kein Spiegeln!)
            A.Rotate(limit=MAX_ANGLE, border_mode=cv2.BORDER_CONSTANT,
                     rotate_method="largest_box", p=0.7),              # schraeg liegende Karten (Box umschliesst ganze Karte)
            A.BBoxSafeRandomCrop(erosion_rate=0.0, p=0.3),             # Zuschneiden, alle Karten bleiben drin
            A.Affine(scale=(0.7, 1.15), translate_percent=(-0.05, 0.05),
                     border_mode=cv2.BORDER_CONSTANT, p=0.5),          # Skalieren (naeher/weiter weg)
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.HueSaturationValue(hue_shift_limit=3, sat_shift_limit=15, val_shift_limit=0, p=0.3),
            A.GaussNoise(std_range=(0.02, 0.07), p=0.4),               # Kamerarauschen
            A.MotionBlur(blur_limit=(3, 5), p=0.15),                   # leichte Unschaerfe
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.4, clip=True),
        seed=seed,
    )


def read_labels(path):
    boxes, cls = [], []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) == 5:
                cls.append(int(p[0]))
                boxes.append([float(v) for v in p[1:]])
    return boxes, cls


def draw(img, boxes, cls):
    h, w = img.shape[:2]
    out = img.copy()
    for (x, y, bw, bh), c in zip(boxes, cls):
        p1 = (int((x - bw / 2) * w), int((y - bh / 2) * h))
        p2 = (int((x + bw / 2) * w), int((y + bh / 2) * h))
        cv2.rectangle(out, p1, p2, (0, 255, 0), 2)
        cv2.putText(out, CLASSES[int(c)], (p1[0], max(p1[1] - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def main():
    ap = argparse.ArgumentParser(description="Augmentation nur fuer train")
    ap.add_argument("--n", type=int, default=N_VARIANTS, help="Varianten pro Originalbild (Drehbuch: 3-5)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    transform = build_transform(args.seed)

    # alte Varianten loeschen, damit das Skript mehrfach laufen darf
    for f in list(IMG_DIR.glob("*_aug*.*")) + list(LBL_DIR.glob("*_aug*.txt")):
        f.unlink()
    PREVIEW.mkdir(parents=True, exist_ok=True)
    for f in PREVIEW.glob("*.jpg"):
        f.unlink()

    originals = [p for p in list_images(IMG_DIR) if "_aug" not in p.stem]
    if not originals:
        raise SystemExit(f"Keine Trainingsbilder in {IMG_DIR}. Erst split_dataset.py ausfuehren.")
    random.seed(args.seed)
    preview_set = set(random.sample(range(len(originals)), min(N_PREVIEW, len(originals))))
    made, dropped = 0, 0
    for i, img_path in enumerate(originals):
        print(f"\rBild {i + 1}/{len(originals)}", end="", flush=True)
        img = cv2.imread(str(img_path))
        boxes, cls = read_labels(LBL_DIR / f"{img_path.stem}.txt")
        for k in range(1, args.n + 1):
            res = transform(image=img, bboxes=boxes, class_labels=cls)
            dropped += len(cls) - len(res["class_labels"])
            name = f"{img_path.stem}_aug{k}"
            cv2.imwrite(str(IMG_DIR / f"{name}.jpg"), res["image"], [cv2.IMWRITE_JPEG_QUALITY, 92])
            lines = [f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                     for (x, y, w, h), c in zip(res["bboxes"], res["class_labels"])]
            (LBL_DIR / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            made += 1
            if i in preview_set and k == 1:
                cv2.imwrite(str(PREVIEW / f"{name}.jpg"), draw(res["image"], res["bboxes"], res["class_labels"]))
    print(f"\n{len(originals)} Originale -> {made} Varianten erzeugt "
          f"({dropped} Boxen fielen weg, weil die Karte nach Drehen/Zuschneiden zu wenig sichtbar war).")
    print(f"Vorschau mit Boxen: {PREVIEW}  -> bitte Stichprobe ansehen!")
    print("Naechster Schritt: python src/data_prep/count_classes.py")


if __name__ == "__main__":
    main()
