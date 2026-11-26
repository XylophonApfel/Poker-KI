"""Zaehltabelle: Karten pro Klasse in train (Originale / Varianten), val und test.

Schreibt results/zaehltabelle.csv (Abgabe laut Drehbuch Abschnitt 3) und ein
Balkendiagramm results/zaehltabelle.png. Warnt bei Klassen unter dem Zielwert.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASSES, DATASET, RESULTS  # noqa: E402

OUT = RESULTS / "zaehltabelle.csv"
TARGET = 40      # Drehbuch: jede Karte mind. 40 Mal (Originale)


def count(part, augmented=None):
    c, n_img = Counter(), 0
    for txt in (DATASET / "labels" / part).glob("*.txt"):
        is_aug = "_aug" in txt.stem
        if augmented is not None and is_aug != augmented:
            continue
        n_img += 1
        for line in txt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                c[int(line.split()[0])] += 1
    return c, n_img


def main():
    tr_o, n_tr_o = count("train", augmented=False)
    tr_a, n_tr_a = count("train", augmented=True)
    va, n_va = count("val")
    te, n_te = count("test")
    if n_tr_o + n_va + n_te == 0:
        raise SystemExit("Keine Labels in dataset/labels/. Erst split_dataset.py ausfuehren.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")   # Semikolon -> oeffnet sauber im deutschen Excel
        w.writerow(["id", "klasse", "train_original", "train_augmentiert", "val", "test",
                    "originale_gesamt", "gesamt_inkl_augmentiert"])
        for i, name in enumerate(CLASSES):
            orig = tr_o[i] + va[i] + te[i]
            w.writerow([i, name, tr_o[i], tr_a[i], va[i], te[i], orig, orig + tr_a[i]])
        w.writerow([])
        w.writerow(["", "BILDER", n_tr_o, n_tr_a, n_va, n_te, n_tr_o + n_va + n_te, n_tr_o + n_tr_a + n_va + n_te])
        so = sum(tr_o.values()) + sum(va.values()) + sum(te.values())
        w.writerow(["", "KARTEN", sum(tr_o.values()), sum(tr_a.values()), sum(va.values()),
                    sum(te.values()), so, so + sum(tr_a.values())])

    # Balkendiagramm fuer die Doku
    x = range(52)
    fig, ax = plt.subplots(figsize=(18, 5))
    b1 = [tr_o[i] for i in x]
    b2 = [va[i] for i in x]
    b3 = [te[i] for i in x]
    ax.bar(x, b1, label="train (Original)")
    ax.bar(x, b2, bottom=b1, label="val")
    ax.bar(x, b3, bottom=[a + b for a, b in zip(b1, b2)], label="test")
    ax.axhline(TARGET, color="red", ls="--", lw=1, label=f"Ziel {TARGET}")
    ax.set_xticks(list(x)); ax.set_xticklabels(CLASSES, rotation=90, fontsize=8)
    ax.set_ylabel("Anzahl markierter Karten"); ax.set_title("Originale Karten pro Klasse (ohne Augmentation)")
    ax.legend(); plt.tight_layout(); plt.savefig(RESULTS / "zaehltabelle.png", dpi=120); plt.close()

    print(f"Bilder: train {n_tr_o} Originale + {n_tr_a} augmentiert, val {n_va}, test {n_te}")
    low = [(CLASSES[i], tr_o[i] + va[i] + te[i]) for i in range(52) if tr_o[i] + va[i] + te[i] < TARGET]
    if low:
        print(f"Unter Zielwert {TARGET} Originale ({len(low)} Klassen): " + ", ".join(f"{n} ({k})" for n, k in low))
    else:
        print(f"Alle 52 Klassen haben mindestens {TARGET} gelabelte Originale.")
    print(f"Tabelle: {OUT}\nDiagramm: {RESULTS / 'zaehltabelle.png'}")
    print("Naechster Schritt: python src/training/train.py")


if __name__ == "__main__":
    main()
