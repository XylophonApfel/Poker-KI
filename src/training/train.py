"""Transfer Learning: yolov8s.pt mit unserem Kartendatensatz weitertrainieren.

Ergebnis:
  results/train/                 results.csv, Grafiken von Ultralytics, weights/
  results/train/genauigkeit.csv  Accuracy pro Epoche auf den Validierungsdaten
  models/best.pt                 bestes Modell (fuer Evaluation und App)

Aufruf:
  python src/training/train.py                     # Standard (GPU/CPU automatisch)
  python src/training/train.py --epochs 3          # kurzer Probelauf
  python src/training/train.py --batch 4           # falls GPU-Speicher (VRAM) voll laeuft
  python src/training/train.py --device 0          # NVIDIA-GPU erzwingen
  python src/training/train.py --resume            # abgebrochenes Training fortsetzen
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATASET, MODELS, RESULTS, list_images  # noqa: E402

EPOCHS = 120
IMGSZ = 800         # hoehere Aufloesung fuer schaerfere Kartenecken (Vielfaches von 32)
PATIENCE = 40       # bricht ab, wenn sich val 40 Epochen lang nicht verbessert
DEFAULT_MODEL = "yolov8s.pt"
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
    ap.add_argument("--batch", type=int, default=None, help="Standard: 8 mit GPU (6 GB VRAM), 4 auf CPU")
    ap.add_argument("--device", default=None, help="'cpu', '0' (erste GPU) oder leer = automatisch")
    ap.add_argument("--model", default=str(MODELS / DEFAULT_MODEL), help="Startmodell")
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
        train_dir = DATASET / "images" / "train"
        if not train_dir.exists() or not list_images(train_dir):
            raise SystemExit("Keine Trainingsbilder. Erst split_dataset.py und augment.py ausfuehren.")

        model_path = Path(args.model)
        if not model_path.exists():
            # einmalig aus dem Internet laden und nach models/ legen
            model_name = model_path.name
            print(f"{model_path} fehlt - lade {model_name} herunter (einmalig, braucht Internet) ...")
            MODELS.mkdir(exist_ok=True)
            tmp = YOLO(model_name)
            if Path(model_name).exists():
                shutil.move(model_name, MODELS / model_name)
            model_path = MODELS / model_name if (MODELS / model_name).exists() else Path(tmp.ckpt_path)

        gpu = gpu_available()
        device = args.device if args.device is not None else (0 if gpu else "cpu")
        batch = args.batch or (8 if gpu and device != "cpu" else 4)
        print(f"Training auf {'GPU' if gpu and device != 'cpu' else 'CPU'} | Modell {model_path.name} | "
              f"Epochen {args.epochs} | Batch {batch} | Bildgroesse {args.imgsz}")

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