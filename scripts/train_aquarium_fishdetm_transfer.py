"""Correct FishDet-M -> aquarium transfer training.

The source checkpoint is passed directly to Ultralytics. Building a YAML model
and manually loading its state dict leaves ``model.ckpt`` unset, so Ultralytics
rebuilds the model at train time and discards those copied weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / ".ultralytics"))

import yaml
from ultralytics import YOLO


ROOT = Path(os.environ.get("AQUARIUM_ROOT", str(Path(__file__).resolve().parents[1] / "data"))).resolve()
DATA_ROOT = ROOT / "ablation"
WEIGHTS = ROOT / "fishdetm_pretrained"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "outputs" / "1280"
SEED = 20260807

RUNS = {
    "direct_baseline_fishdetm": {
        "checkpoint": WEIGHTS / "fishdetm_yolov8s_baseline_best.pt",
        "data": DATA_ROOT / "yolo7" / "data.yaml",
        "classes": set(range(7)),
    },
    "direct_p2_fishdetm": {
        "checkpoint": WEIGHTS / "fishdetm_yolov8s_p2_best.pt",
        "data": DATA_ROOT / "yolo7" / "data.yaml",
        "classes": set(range(7)),
    },
    "detector_baseline_fishdetm": {
        "checkpoint": WEIGHTS / "fishdetm_yolov8s_baseline_best.pt",
        "data": DATA_ROOT / "yolo_fish" / "data.yaml",
        "classes": {0},
    },
    "detector_p2_fishdetm": {
        "checkpoint": WEIGHTS / "fishdetm_yolov8s_p2_best.pt",
        "data": DATA_ROOT / "yolo_fish" / "data.yaml",
        "classes": {0},
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_dataset(data_yaml: Path, allowed_classes: set[int]) -> dict:
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML missing: {data_yaml}")
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(config["path"])
    found_classes: set[int] = set()
    image_count = label_count = box_count = 0
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for split in ("train", "val", "test"):
        image_dir = dataset_root / config[split]
        label_dir = Path(str(image_dir).replace("images", "labels"))
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Dataset split missing: {image_dir} or {label_dir}")
        image_count += sum(p.suffix.lower() in extensions for p in image_dir.iterdir())
        for label in label_dir.glob("*.txt"):
            label_count += 1
            for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"Malformed label {label}:{line_number}")
                class_id = int(fields[0])
                found_classes.add(class_id)
                box_count += 1
    invalid = found_classes - allowed_classes
    if invalid:
        raise ValueError(f"Unexpected classes in {data_yaml}: {sorted(invalid)}")
    return {
        "yaml": str(data_yaml),
        "images": image_count,
        "labels": label_count,
        "boxes": box_count,
        "classes": sorted(found_classes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--run", action="append", choices=tuple(RUNS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = args.run or list(RUNS)
    manifest = {
        "status": "PREFLIGHT" if args.dry_run else "RUNNING",
        "seed": SEED,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "patience": args.patience,
        "output": str(output),
        "runs": [],
    }

    for run_name in selected:
        spec = RUNS[run_name]
        checkpoint = spec["checkpoint"]
        if not checkpoint.exists():
            raise FileNotFoundError(f"FishDet-M checkpoint missing: {checkpoint}")
        dataset = validate_dataset(spec["data"], spec["classes"])
        model = YOLO(str(checkpoint))
        source_nc = int(model.model.yaml["nc"])
        source_names = model.model.names
        if source_nc != 1 or list(source_names.values()) != ["fish"]:
            raise ValueError(
                f"Not a one-class FishDet-M checkpoint: {checkpoint}; "
                f"nc={source_nc}, names={source_names}"
            )
        item = {
            "name": run_name,
            "source_checkpoint": str(checkpoint),
            "source_sha256": sha256(checkpoint),
            "source_nc": source_nc,
            "source_names": source_names,
            "dataset": dataset,
            "best": str(output / "runs" / run_name / "weights" / "best.pt"),
        }
        manifest["runs"].append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
        if args.dry_run:
            continue

        best = Path(item["best"])
        if best.exists():
            print(f"[skip] completed run already exists: {best}", flush=True)
            continue
        model.train(
            data=str(spec["data"]),
            project=str(output / "runs"),
            name=run_name,
            # Reuse an incomplete run directory after an interrupted local run.
            # Completed runs are skipped above when best.pt exists.
            exist_ok=True,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            seed=SEED,
            patience=args.patience,
            plots=False,
            verbose=True,
        )

    manifest["status"] = "PREFLIGHT_OK" if args.dry_run else "COMPLETE"
    (output / "transfer_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[{manifest['status']}] artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
