"""Transfer Learning: yolov8n.pt mit unserem Kartendatensatz weitertrainieren.

Ergebnis:
  results/train/                 results.csv, Grafiken von Ultralytics, weights/
  results/train/genauigkeit.csv  Accuracy pro Epoche auf den Validierungsdaten
  models/best.pt                 bestes Modell (fuer Evaluation und App)

Aufruf:
  python src/training/train.py                     # Standard (CPU oder GPU automatisch)
  python src/training/train.py --epochs 3          # kurzer Probelauf
  python src/training/train.py --device 0          # NVIDIA-GPU erzwingen
  python src/training/train.py --resume            # abgebrochenes Training fortsetzen
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATASET, MODELS, RESULTS  # noqa: E402

EPOCHS = 80
IMGSZ = 640
PATIENCE = 20       # bricht ab, wenn sich val 20 Epochen lang nicht verbessert
ACC_CSV = RESULTS / "train" / "genauigkeit.csv"


def gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def log_accuracy(trainer):
    """Nach jeder Epoche: Anteil richtig erkannter Karten auf val (aus der Konfusionsmatrix)."""
    if trainer.epoch >= trainer.epochs:      # Schlussvalidierung, keine neue Epoche
        return
    try:
        m = trainer.validator.confusion_matrix.matrix      # Zeilen = erkannt, Spalten = wahr, letzte = Hintergrund
        nc = m.shape[0] - 1
        total = m[:, :nc].sum()
        acc = float(m.diagonal()[:nc].sum() / total) if total else 0.0
        new = not ACC_CSV.exists() or trainer.epoch == 0
        with open(ACC_CSV, "w" if new else "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["epoch", "val_accuracy"])
            w.writerow([trainer.epoch + 1, f"{acc:.4f}"])
    except Exception as e:  # Kurve ist nur ein Extra - Training nie deswegen abbrechen
        print(f"(Accuracy-Kurve nicht verfuegbar: {e})")


def main():
    ap = argparse.ArgumentParser(description="YOLOv8-Training")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--batch", type=int, default=None, help="Standard: 16 mit GPU, 8 auf CPU (bei 'out of memory' kleiner)")
    ap.add_argument("--device", default=None, help="'cpu', '0' (erste GPU) oder leer = automatisch")
    ap.add_argument("--model", default=str(MODELS / "yolov8n.pt"), help="Startmodell")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true", help="abgebrochenes Training fortsetzen")
    args = ap.parse_args()

    from ultralytics import YOLO

    if args.resume:
        last = RESULTS / "train" / "weights" / "last.pt"
        if not last.exists():
            raise SystemExit(f"{last} fehlt - nichts zum Fortsetzen.")
        model = YOLO(str(last))
        model.add_callback("on_fit_epoch_end", log_accuracy)
        model.train(resume=True)
    else:
        data = DATASET / "data.yaml"
        if not (DATASET / "images" / "train").exists() or not any((DATASET / "images" / "train").glob("*.jpg")):
            raise SystemExit("Keine Trainingsbilder. Erst split_dataset.py und augment.py ausfuehren.")
        model_path = Path(args.model)
        if not model_path.exists():
            # einmalig aus dem Internet laden und nach models/ legen
            print(f"{model_path} fehlt - lade yolov8n.pt herunter (einmalig, braucht Internet) ...")
            MODELS.mkdir(exist_ok=True)
            tmp = YOLO("yolov8n.pt")
            if Path("yolov8n.pt").exists():
                shutil.move("yolov8n.pt", MODELS / "yolov8n.pt")
            model_path = MODELS / "yolov8n.pt" if (MODELS / "yolov8n.pt").exists() else Path(tmp.ckpt_path)

        gpu = gpu_available()
        device = args.device if args.device is not None else (0 if gpu else "cpu")
        batch = args.batch or (16 if gpu else 8)
        print(f"Training auf {'GPU' if gpu and device != 'cpu' else 'CPU'} | Epochen {args.epochs} | "
              f"Batch {batch} | Bildgroesse {args.imgsz}")

        if ACC_CSV.exists():
            ACC_CSV.unlink()
        model = YOLO(str(model_path))
        model.add_callback("on_fit_epoch_end", log_accuracy)
        model.train(
            data=str(data.resolve()),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=batch,
            device=device,
            patience=PATIENCE,
            project=str(RESULTS),
            name="train",
            exist_ok=True,
            workers=args.workers,      # Windows: klein halten
            seed=42,
            plots=True,
            # eingebaute YOLO-Augmentation an das Drehbuch anpassen:
            fliplr=0.0,                # KEIN Spiegeln (Standard waere 0.5!)
            flipud=0.0,
            hsv_h=0.0,                 # keine Farbtonverschiebung (rot/schwarz muss bleiben)
        )

    best = RESULTS / "train" / "weights" / "best.pt"
    MODELS.mkdir(exist_ok=True)
    shutil.copy2(best, MODELS / "best.pt")
    print(f"\nFertig. Bestes Modell: {MODELS / 'best.pt'}")
    print("Naechster Schritt: python src/training/evaluate.py")


if __name__ == "__main__":   # unter Windows zwingend noetig (Multiprocessing)
    main()
