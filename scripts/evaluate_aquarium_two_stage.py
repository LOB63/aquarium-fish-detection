"""Evaluate the only two relevant aquarium systems on the held-out test set.

Both arms share the same trained EfficientNet-B0 classifier.  The controlled
variable is the FishDet-M-pretrained detector: YOLOv8s baseline vs YOLOv8s-P2.
"""
from __future__ import annotations

import json
import os
import argparse
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / ".ultralytics"))

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from torch import nn
from ultralytics import YOLO


PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("AQUARIUM_ROOT", str(PROJECT / "data"))).resolve() / "ablation"
RUNS = PROJECT / "outputs" / "1280" / "runs"
CLASSIFIER_CKPT = ROOT / "runs" / "classifier" / "best.pt"
OUTPUT = PROJECT / "outputs" / "1280" / "evaluation"
CLASS_NAMES = [
    "clownfish", "blue_tang", "sailfin_tang", "blue_damselfish",
    "yellowtail_blue_damselfish", "anthias", "other_fish",
]


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
    return inter / union if union else 0.0


def ap101(tp, fp, positives):
    if positives == 0:
        return None
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, 1e-12)
    return float(np.mean([np.max(precision[recall >= r], initial=0.0) for r in np.linspace(0, 1, 101)]))


def load_ground_truth():
    image_dir = ROOT / "yolo7" / "images" / "test"
    label_dir = ROOT / "yolo7" / "labels" / "test"
    gt = defaultdict(list)
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    relative_areas = []
    for path in images:
        with Image.open(path) as im:
            width, height = im.size
        # Match Ultralytics' label-cache behaviour: exact duplicate rows are
        # removed (the test set currently contains one duplicate annotation).
        rows = (label_dir / f"{path.stem}.txt").read_text(encoding="utf-8-sig").splitlines()
        # A legacy annotation contains the exact same box as both a concrete
        # species and ``other_fish``. Keep the more specific species label.
        parsed = [line.split() for line in rows if line.strip()]
        by_box = {}
        for fields in parsed:
            c, cx, cy, w, h = fields
            key = tuple(round(float(v), 8) for v in (cx, cy, w, h))
            previous = by_box.get(key)
            if previous is None or (int(previous[0]) == 6 and int(c) != 6):
                by_box[key] = fields
        for c, cx, cy, w, h in by_box.values():
            cx, cy, w, h = map(float, (cx, cy, w, h))
            box = [(cx-w/2)*width, (cy-h/2)*height, (cx+w/2)*width, (cy+h/2)*height]
            area = w * h
            relative_areas.append(area)
            gt[path.name].append({"class": int(c), "box": box, "area": area})
    # Dataset-specific small-object threshold fixed from the original analysis.
    small_threshold = 0.0050525119347592
    return images, dict(gt), small_threshold, relative_areas


def predict_arm(detector_path, classifier, transform, class_map, images, device, imgsz):
    detector = YOLO(str(detector_path))
    output = defaultdict(list)
    for index, path in enumerate(images, 1):
        result = detector.predict(str(path), conf=0.001, iou=0.7, imgsz=imgsz,
                                  max_det=500, device=device, verbose=False)[0]
        with Image.open(path) as source:
            image = source.convert("RGB")
        tensors, meta = [], []
        for item in result.boxes:
            x1, y1, x2, y2 = item.xyxy[0].tolist()
            mx, my = 0.05 * (x2-x1), 0.05 * (y2-y1)
            crop = image.crop((max(0, x1-mx), max(0, y1-my), min(image.width, x2+mx), min(image.height, y2+my)))
            tensors.append(transform(crop))
            meta.append((float(item.conf.item()), [x1, y1, x2, y2]))
        for start in range(0, len(tensors), 128):
            batch = torch.stack(tensors[start:start+128]).to(device)
            with torch.no_grad():
                probs = torch.softmax(classifier(batch), dim=1).cpu()
            for prob, (det_score, box) in zip(probs, meta[start:start+128]):
                local = int(prob.argmax())
                cls = class_map[local]
                output[path.name].append({"class": cls, "box": box, "score": det_score * float(prob[local])})
        if index % 10 == 0 or index == len(images):
            print(f"{detector_path.parent.parent.name}: {index}/{len(images)}", flush=True)
    return dict(output), sum(p.numel() for p in detector.model.parameters())


def predict_direct(detector_path, images, device, imgsz):
    """Predictions from a direct seven-class detector (A or B)."""
    detector = YOLO(str(detector_path))
    output = defaultdict(list)
    for index, path in enumerate(images, 1):
        result = detector.predict(str(path), conf=0.001, iou=0.7, imgsz=imgsz,
                                  max_det=500, device=device, verbose=False)[0]
        for item in result.boxes:
            output[path.name].append({
                "class": int(item.cls.item()),
                "box": item.xyxy[0].tolist(),
                "score": float(item.conf.item()),
            })
        if index % 10 == 0 or index == len(images):
            print(f"{detector_path.parent.parent.name}: {index}/{len(images)}", flush=True)
    return dict(output), sum(p.numel() for p in detector.model.parameters())


def evaluate(predictions, gt, small_threshold):
    thresholds = [0.5 + 0.05*i for i in range(10)]
    per_class = {}
    for cls in range(len(CLASS_NAMES)):
        preds = sorted([p | {"image": image} for image, ps in predictions.items() for p in ps if p["class"] == cls], key=lambda x: x["score"], reverse=True)
        positives = sum(g["class"] == cls for gs in gt.values() for g in gs)
        aps = []
        for threshold in thresholds:
            matched = {image: set() for image in gt}
            tp, fp = [], []
            for pred in preds:
                candidates = [(i, iou(pred["box"], g["box"])) for i, g in enumerate(gt[pred["image"]]) if g["class"] == cls and i not in matched[pred["image"]]]
                best = max(candidates, key=lambda x: x[1]) if candidates else None
                if best and best[1] >= threshold:
                    matched[pred["image"]].add(best[0]); tp.append(1); fp.append(0)
                else:
                    tp.append(0); fp.append(1)
            aps.append(ap101(tp, fp, positives))
        per_class[CLASS_NAMES[cls]] = {"AP50": aps[0], "AP50_95": float(np.mean(aps)), "gt": positives}

    # Size-aware, class-aware recall at IoU=.5. This directly measures whether
    # P2 recovers more small fish without conflating it with classifier score.
    def matched_recall(area_filter):
        hit = total = 0
        for image, objects in gt.items():
            used = set()
            for g in objects:
                if not area_filter(g["area"]):
                    continue
                total += 1
                candidates = [(i, iou(p["box"], g["box"])) for i, p in enumerate(predictions.get(image, [])) if p["class"] == g["class"] and i not in used]
                best = max(candidates, key=lambda x: x[1]) if candidates else None
                if best and best[1] >= 0.5:
                    used.add(best[0]); hit += 1
        return {"matched": hit, "gt": total, "recall": hit / total if total else 0.0}

    return {
        "mAP50": float(np.mean([x["AP50"] for x in per_class.values()])),
        "mAP50_95": float(np.mean([x["AP50_95"] for x in per_class.values()])),
        "per_class": per_class,
        "small_recall50": matched_recall(lambda a: a < small_threshold),
        "non_small_recall50": matched_recall(lambda a: a >= small_threshold),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()
    runs = args.runs.expanduser().resolve()
    output = args.output.expanduser().resolve()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(CLASSIFIER_CKPT, map_location=device, weights_only=False)
    if ckpt["classes"] != CLASS_NAMES:
        raise ValueError(f"Classifier classes mismatch: {ckpt['classes']}")
    classifier = models.efficientnet_b0(weights=None)
    classifier.classifier[1] = nn.Linear(classifier.classifier[1].in_features, len(CLASS_NAMES))
    classifier.load_state_dict(ckpt["state_dict"])
    classifier.to(device).eval()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    images, gt, small_threshold, areas = load_ground_truth()
    direct_arms = {
        "A_direct_baseline": runs / "direct_baseline_fishdetm" / "weights" / "best.pt",
        "B_direct_p2": runs / "direct_p2_fishdetm" / "weights" / "best.pt",
    }
    two_stage_arms = {
        "baseline_plus_efficientnet_b0": runs / "detector_baseline_fishdetm" / "weights" / "best.pt",
        "p2_plus_efficientnet_b0": runs / "detector_p2_fishdetm" / "weights" / "best.pt",
    }
    report = {"imgsz": args.imgsz, "test_images": len(images), "test_instances": sum(map(len, gt.values())), "small_threshold_relative_area": small_threshold, "classifier": str(CLASSIFIER_CKPT), "arms": {}}
    for name, detector in direct_arms.items():
        predictions, parameters = predict_direct(detector, images, device, args.imgsz)
        report["arms"][name] = evaluate(predictions, gt, small_threshold) | {"detector": str(detector), "detector_parameters": parameters}
    for name, detector in two_stage_arms.items():
        predictions, parameters = predict_arm(detector, classifier, transform, {i:i for i in range(7)}, images, device, args.imgsz)
        report["arms"][name] = evaluate(predictions, gt, small_threshold) | {"detector": str(detector), "detector_parameters": parameters}
    a = report["arms"]["A_direct_baseline"]
    b = report["arms"]["B_direct_p2"]
    c = report["arms"]["baseline_plus_efficientnet_b0"]
    d = report["arms"]["p2_plus_efficientnet_b0"]
    def delta(right, left):
        return {
            "mAP50": right["mAP50"] - left["mAP50"],
            "mAP50_95": right["mAP50_95"] - left["mAP50_95"],
            "small_recall50": right["small_recall50"]["recall"] - left["small_recall50"]["recall"],
        }
    report["contrasts"] = {
        "B_minus_A__P2_in_direct": delta(b, a),
        "C_minus_A__decoupling_without_P2": delta(c, a),
        "D_minus_C__P2_in_two_stage": delta(d, c),
        "D_minus_B__decoupling_with_P2": delta(d, b),
    }
    output.mkdir(parents=True, exist_ok=True)
    target = output / "two_stage_test_metrics.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"[COMPLETE] {target}", flush=True)


if __name__ == "__main__":
    main()
