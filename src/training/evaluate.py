"""Evaluation auf den Testdaten (Lastenheft 6.3). Alles landet in results/evaluation/.

1. Jede erkannte Karte wird der passenden markierten Karte zugeordnet (IoU >= 0.5).
   Nicht gefundene Karten -> "nicht_erkannt", erkannte Karten ohne echte Karte -> "fehlalarm".
2. Accuracy, Precision, Recall, F1 pro Klasse und gesamt (scikit-learn)
   -> kennzahlen_pro_klasse.csv, classification_report.txt
3. Konfusionsmatrix: alle 52 Klassen, verdichtet nach Farbe und nach Wert
   -> konfusionsmatrix_52.png, konfusionsmatrix_farbe.png, konfusionsmatrix_wert.png
4. Trainingsverlauf: Loss und Accuracy/Precision/Recall/mAP pro Epoche
   -> trainingsverlauf.png
5. Zusaetzlich die Standard-Auswertung von Ultralytics auf test (mAP50, mAP50-95)
   -> yolo_test/

Aufruf:
  python src/training/evaluate.py
  python src/training/evaluate.py --conf 0.5      # andere Sicherheitsschwelle
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATASET, MODELS, RESULTS, list_images  # noqa: E402

TEST_IMG = DATASET / "images" / "test"
TEST_LBL = DATASET / "labels" / "test"
OUT = RESULTS / "evaluation"
TRAIN_DIR = RESULTS / "train"

CONF = 0.6     # gleiche Schwelle wie spaeter in der App (Lastenheft 5.3)
IOU = 0.5
MISS, FP = "nicht_erkannt", "fehlalarm"


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0


def match_image(gt, pred, names):
    """gt/pred: Listen von (klasse, [x1,y1,x2,y2]). Gibt Paare (wahr, erkannt) zurueck."""
    pairs = sorted(((iou(g[1], p[1]), gi, pi) for gi, g in enumerate(gt) for pi, p in enumerate(pred)), reverse=True)
    used_g, used_p, out = set(), set(), []
    for v, gi, pi in pairs:
        if v < IOU:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi); used_p.add(pi)
        out.append((names[gt[gi][0]], names[pred[pi][0]]))
    out += [(names[g[0]], MISS) for gi, g in enumerate(gt) if gi not in used_g]
    out += [(FP, names[p[0]]) for pi, p in enumerate(pred) if pi not in used_p]
    return out


def plot_cm(y_true, y_pred, labels, path, title, size):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(size, size * 0.85))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False,
                annot_kws={"size": 6 if size > 15 else 9}, mask=(cm == 0) if size > 15 else None,
                linewidths=0.3 if size > 15 else 0, linecolor="#eeeeee")
    plt.xlabel("erkannt"); plt.ylabel("wahr"); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return cm


def reduce(label, part):
    if label in (MISS, FP):
        return label
    farbe, wert = label.split("_")
    return farbe if part == "farbe" else wert


def plot_training():
    csv_path = TRAIN_DIR / "results.csv"
    if not csv_path.exists():
        print("Kein results/train/results.csv gefunden, Trainingsverlauf wird uebersprungen.")
        return
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    acc_path = TRAIN_DIR / "genauigkeit.csv"
    acc = pd.read_csv(acc_path) if acc_path.exists() else None

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for kind, a, title in [("box_loss", ax[0], "Box-Loss (Lage der Box)"),
                           ("cls_loss", ax[1], "Klassifikations-Loss (welche Karte)")]:
        a.plot(df["epoch"], df[f"train/{kind}"], label="Training")
        a.plot(df["epoch"], df[f"val/{kind}"], label="Validierung")
        a.set_title(title); a.set_xlabel("Epoche"); a.set_ylabel("Loss"); a.legend(); a.grid(alpha=0.3)
    if acc is not None and len(acc):
        ax[2].plot(acc["epoch"], acc["val_accuracy"], label="Accuracy", lw=2.2, color="black")
    for col, lab in [("metrics/precision(B)", "Precision"), ("metrics/recall(B)", "Recall"),
                     ("metrics/mAP50(B)", "mAP50")]:
        if col in df:
            ax[2].plot(df["epoch"], df[col], label=lab)
    ax[2].set_title("Validierung: Accuracy / Precision / Recall / mAP50"); ax[2].set_xlabel("Epoche")
    ax[2].set_ylim(0, 1); ax[2].legend(); ax[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT / "trainingsverlauf.png", dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser(description="Evaluation auf den Testdaten")
    ap.add_argument("--conf", type=float, default=CONF, help="Mindest-Sicherheit einer Erkennung")
    ap.add_argument("--model", default=str(MODELS / "best.pt"))
    ap.add_argument("--ohne-yolo-val", action="store_true", help="Ultralytics-Auswertung (mAP) ueberspringen")
    args = ap.parse_args()

    from ultralytics import YOLO

    OUT.mkdir(parents=True, exist_ok=True)
    if not Path(args.model).exists():
        raise SystemExit(f"{args.model} fehlt. Erst trainieren (src/training/train.py).")
    model = YOLO(args.model)
    names = model.names                      # {0: 'kreuz_2', ...}
    classes = [names[i] for i in range(len(names))]

    y_true, y_pred = [], []
    images = list_images(TEST_IMG)
    if not images:
        raise SystemExit(f"Keine Testbilder in {TEST_IMG}.")
    for n, img_path in enumerate(images, 1):
        print(f"\rTestbild {n}/{len(images)}", end="", flush=True)
        res = model.predict(str(img_path), conf=args.conf, verbose=False)[0]
        h, w = res.orig_shape
        gt = []
        lbl = TEST_LBL / f"{img_path.stem}.txt"
        if lbl.exists():
            for line in lbl.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) == 5:
                    c, x, y, bw, bh = int(p[0]), *map(float, p[1:])
                    gt.append((c, [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]))
        pred = [(int(c), b.tolist()) for c, b in zip(res.boxes.cls, res.boxes.xyxy)]
        for t, p in match_image(gt, pred, names):
            y_true.append(t); y_pred.append(p)
    print()

    # --- Kennzahlen ---
    rep_txt = classification_report(y_true, y_pred, labels=classes, zero_division=0, digits=3)
    rep = classification_report(y_true, y_pred, labels=classes, zero_division=0, output_dict=True)
    real = [(t, p) for t, p in zip(y_true, y_pred) if t != FP]          # nur echte Karten
    acc = accuracy_score([t for t, _ in real], [p for _, p in real]) if real else 0.0
    rows = []
    for c in classes:
        n = sum(1 for t, _ in real if t == c)
        ok = sum(1 for t, p in real if t == c and p == c)
        r = rep[c]
        rows.append([c, n, ok / n if n else 0, r["precision"], r["recall"], r["f1-score"]])
    df = pd.DataFrame(rows, columns=["klasse", "anzahl_test", "accuracy", "precision", "recall", "f1"])
    df.to_csv(OUT / "kennzahlen_pro_klasse.csv", sep=";", decimal=",", index=False,
              float_format="%.3f", encoding="utf-8-sig")

    n_fp = sum(1 for t in y_true if t == FP)
    n_miss = sum(1 for p in y_pred if p == MISS)
    summary = (f"Modell: {args.model} | Schwelle conf={args.conf} | IoU={IOU}\n"
               f"Testbilder: {len(images)} | markierte Karten: {len(real)} | "
               f"nicht erkannt: {n_miss} | Fehlalarme: {n_fp}\n"
               f"Accuracy gesamt (richtig erkannte / alle markierten Karten): {acc:.3f}\n"
               f"Precision makro: {rep['macro avg']['precision']:.3f} | Recall makro: {rep['macro avg']['recall']:.3f} | "
               f"F1 makro: {rep['macro avg']['f1-score']:.3f}\n")

    # --- Ultralytics-Auswertung auf test (mAP) ---
    if not args.ohne_yolo_val:
        try:
            m = model.val(data=str((DATASET / "data.yaml").resolve()), split="test", conf=0.001,
                          project=str(OUT), name="yolo_test", exist_ok=True, plots=True, verbose=False)
            summary += f"mAP50 (test): {m.box.map50:.3f} | mAP50-95 (test): {m.box.map:.3f}\n"
        except Exception as e:
            print(f"(Ultralytics-Auswertung uebersprungen: {e})")

    (OUT / "classification_report.txt").write_text(summary + "\n" + rep_txt, encoding="utf-8")
    print(summary + "\n" + rep_txt)

    # --- Konfusionsmatrizen ---
    plot_cm(y_true, y_pred, classes + [MISS, FP], OUT / "konfusionsmatrix_52.png", "Konfusionsmatrix (alle Karten)", 26)
    for part in ("farbe", "wert"):
        order = [reduce(c, part) for c in classes]
        labs = sorted(set(order), key=order.index)
        plot_cm([reduce(t, part) for t in y_true], [reduce(p, part) for p in y_pred], labs + [MISS, FP],
                OUT / f"konfusionsmatrix_{part}.png", f"Konfusionsmatrix nach {part.capitalize()}",
                8 if part == "farbe" else 12)

    # --- haeufigste Verwechslungen (fuer die Interpretation) ---
    conf = Counter((t, p) for t, p in zip(y_true, y_pred) if t != p and t != FP and p != MISS)
    lines = [f"{t} -> {p}: {n}x" for (t, p), n in conf.most_common(15)]
    (OUT / "verwechslungen.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Haeufigste Verwechslungen:\n" + ("\n".join(lines) or "keine"))

    plot_training()
    print(f"\nAlle Ergebnisse in {OUT}")


if __name__ == "__main__":
    main()
