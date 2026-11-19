"""Re-evaluate all final aquarium arms with fixed relative-area AP strata.

The cut points are computed from the *training* labels only (25th and 75th
percentiles of bbox_area / image_area) and then frozen for both the 1280 and
640 test evaluations.  This avoids selecting thresholds on the test set and
keeps each ground-truth fish in the same stratum across input resolutions.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / ".ultralytics"))

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import nn
from torchvision import models, transforms
from ultralytics import YOLO

import evaluate_aquarium_two_stage as common


WORKSPACE = Path(__file__).resolve().parents[1]
DATA_ROOT = common.ROOT / "yolo7"
CLASSIFIER_CKPT = common.CLASSIFIER_CKPT
OUTPUT = WORKSPACE / "outputs" / "scale_ap"
RUN_SETS = {
    "1280": WORKSPACE / "outputs" / "1280" / "runs",
    "640": WORKSPACE / "outputs" / "640" / "runs",
}
ARM_PATHS = {
    "A_direct_baseline": ("direct_baseline_fishdetm", "direct"),
    "B_direct_p2": ("direct_p2_fishdetm", "direct"),
    "C_baseline_plus_efficientnet_b0": ("detector_baseline_fishdetm", "two_stage"),
    "D_p2_plus_efficientnet_b0": ("detector_p2_fishdetm", "two_stage"),
}


def unique_label_rows(label_path: Path):
    """Use the same duplicate/conflict policy as the final test evaluator."""
    by_box = {}
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        c, cx, cy, w, h = fields
        key = tuple(round(float(value), 8) for value in (cx, cy, w, h))
        previous = by_box.get(key)
        if previous is None or (int(previous[0]) == 6 and int(c) != 6):
            by_box[key] = fields
    return list(by_box.values())


def training_cut_points():
    areas = []
    for label in sorted((DATA_ROOT / "labels" / "train").glob("*.txt")):
        for _, _, _, width, height in unique_label_rows(label):
            areas.append(float(width) * float(height))
    q25, q75 = (float(v) for v in np.quantile(np.asarray(areas), [0.25, 0.75]))
    return q25, q75, len(areas)


def load_classifier(device):
    checkpoint = torch.load(CLASSIFIER_CKPT, map_location=device, weights_only=False)
    if checkpoint["classes"] != common.CLASS_NAMES:
        raise ValueError(f"Classifier classes mismatch: {checkpoint['classes']}")
    classifier = models.efficientnet_b0(weights=None)
    classifier.classifier[1] = nn.Linear(classifier.classifier[1].in_features, len(common.CLASS_NAMES))
    classifier.load_state_dict(checkpoint["state_dict"])
    classifier.to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    return classifier, transform


def predict_two_stage(detector_path, classifier, transform, images, device, imgsz):
    """Return both end-to-end seven-class and raw one-class predictions."""
    detector = YOLO(str(detector_path))
    system = defaultdict(list)
    detector_only = defaultdict(list)
    for index, path in enumerate(images, 1):
        result = detector.predict(
            str(path), conf=0.001, iou=0.7, imgsz=imgsz,
            max_det=500, device=device, verbose=False,
        )[0]
        with Image.open(path) as source:
            image = source.convert("RGB")
        tensors, metadata = [], []
        for item in result.boxes:
            x1, y1, x2, y2 = item.xyxy[0].tolist()
            det_score = float(item.conf.item())
            box = [x1, y1, x2, y2]
            detector_only[path.name].append({"class": 0, "box": box, "score": det_score})
            margin_x, margin_y = .05 * (x2 - x1), .05 * (y2 - y1)
            crop = image.crop((
                max(0, x1 - margin_x), max(0, y1 - margin_y),
                min(image.width, x2 + margin_x), min(image.height, y2 + margin_y),
            ))
            tensors.append(transform(crop))
            metadata.append((det_score, box))
        for start in range(0, len(tensors), 128):
            batch = torch.stack(tensors[start:start + 128]).to(device)
            with torch.no_grad():
                probabilities = torch.softmax(classifier(batch), dim=1).cpu()
            for probability, (det_score, box) in zip(probabilities, metadata[start:start + 128]):
                cls = int(probability.argmax())
                system[path.name].append({
                    "class": cls,
                    "box": box,
                    "score": det_score * float(probability[cls]),
                })
        if index % 10 == 0 or index == len(images):
            print(f"{imgsz} {detector_path.parent.parent.name}: {index}/{len(images)}", flush=True)
    return dict(system), dict(detector_only)


def make_coco(images, gt, *, detector_only=False):
    coco = COCO()
    image_rows, annotations = [], []
    image_ids = {}
    annotation_id = 1
    for image_id, path in enumerate(images, 1):
        with Image.open(path) as source:
            width, height = source.size
        image_ids[path.name] = image_id
        image_rows.append({"id": image_id, "file_name": path.name, "width": width, "height": height})
        for item in gt[path.name]:
            x1, y1, x2, y2 = item["box"]
            category_id = 0 if detector_only else item["class"]
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
            })
            annotation_id += 1
    categories = ([{"id": 0, "name": "fish"}] if detector_only else
                  [{"id": i, "name": name} for i, name in enumerate(common.CLASS_NAMES)])
    coco.dataset = {"images": image_rows, "annotations": annotations, "categories": categories}
    coco.createIndex()
    return coco, image_ids


def coco_predictions(predictions, image_ids, *, detector_only=False):
    rows = []
    for image_name, items in predictions.items():
        for item in items:
            x1, y1, x2, y2 = item["box"]
            rows.append({
                "image_id": image_ids[image_name],
                "category_id": 0 if detector_only else item["class"],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": item["score"],
            })
    return rows


def area_ap(coco_gt, prediction_rows, low, high):
    coco_dt = coco_gt.loadRes(prediction_rows)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.maxDets = [1, 100, 500]
    evaluator.params.areaRng = [[low, high]]
    evaluator.params.areaRngLbl = ["stratum"]
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"][:, :, :, 0, 2]
    valid = precision > -1
    overall = float(precision[valid].mean()) if valid.any() else None
    at_50 = precision[0]
    valid_50 = at_50 > -1
    ap50 = float(at_50[valid_50].mean()) if valid_50.any() else None
    per_class = {}
    for category_index, category_id in enumerate(evaluator.params.catIds):
        category_precision = precision[:, :, category_index]
        category_valid = category_precision > -1
        category_at_50 = category_precision[0]
        category_valid_50 = category_at_50 > -1
        per_class[coco_gt.cats[category_id]["name"]] = {
            "AP50": (
                float(category_at_50[category_valid_50].mean())
                if category_valid_50.any() else None
            ),
            "AP50_95": (
                float(category_precision[category_valid].mean())
                if category_valid.any() else None
            ),
        }
    return {"AP50": ap50, "AP50_95": overall, "per_class": per_class}


def evaluate_prediction_set(predictions, coco_gt, image_ids, strata, *, detector_only=False):
    rows = coco_predictions(predictions, image_ids, detector_only=detector_only)
    return {name: area_ap(coco_gt, rows, low, high) for name, low, high in strata}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache_root = OUTPUT / "predictions"
    cache_root.mkdir(exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    classifier, transform = load_classifier(device)
    images, gt, _, _ = common.load_ground_truth()
    image_areas = []
    for path in images:
        with Image.open(path) as source:
            image_areas.append(source.width * source.height)
    if len(set(image_areas)) != 1:
        raise ValueError("Relative-area COCO conversion currently requires a common image size")
    image_area = image_areas[0]
    q25, q75, training_instances = training_cut_points()
    strata = [
        ("relative_small", 0.0, q25 * image_area),
        ("relative_medium", q25 * image_area, q75 * image_area),
        ("relative_large", q75 * image_area, 1e12),
        ("all", 0.0, 1e12),
    ]
    coco_system, image_ids = make_coco(images, gt, detector_only=False)
    coco_detector, detector_image_ids = make_coco(images, gt, detector_only=True)
    report = {
        "method": "COCOeval 101-point AP, IoU=0.50:0.95, maxDets=500",
        "threshold_source": "training-set relative bbox-area 25th and 75th percentiles",
        "train_instances_used_for_thresholds": training_instances,
        "relative_area_cut_points": {"q25": q25, "q75": q75},
        "test_images": len(images),
        "test_instances": sum(len(items) for items in gt.values()),
        "test_stratum_counts": {},
        "test_stratum_class_counts": {},
        "resolutions": {},
    }
    for name, low, high in strata[:-1]:
        report["test_stratum_counts"][name] = sum(
            low <= item["area"] * image_area < high
            for items in gt.values() for item in items
        )
        report["test_stratum_class_counts"][name] = {
            class_name: sum(
                item["class"] == class_id and low <= item["area"] * image_area < high
                for items in gt.values() for item in items
            )
            for class_id, class_name in enumerate(common.CLASS_NAMES)
        }

    for size_text, runs in RUN_SETS.items():
        imgsz = int(size_text)
        size_report = {"arms": {}, "detector_only": {}}
        for arm, (run_name, mode) in ARM_PATHS.items():
            checkpoint = runs / run_name / "weights" / "best.pt"
            cache = cache_root / f"{imgsz}_{arm}.json"
            if cache.exists():
                payload = json.loads(cache.read_text(encoding="utf-8"))
                predictions = payload["system"]
                detector_predictions = payload.get("detector_only")
            elif mode == "direct":
                predictions, _ = common.predict_direct(checkpoint, images, device, imgsz)
                detector_predictions = None
                cache.write_text(json.dumps({"system": predictions}), encoding="utf-8")
            else:
                predictions, detector_predictions = predict_two_stage(
                    checkpoint, classifier, transform, images, device, imgsz,
                )
                cache.write_text(json.dumps({
                    "system": predictions,
                    "detector_only": detector_predictions,
                }), encoding="utf-8")
            size_report["arms"][arm] = evaluate_prediction_set(
                predictions, coco_system, image_ids, strata,
            )
            if detector_predictions is not None:
                size_report["detector_only"][arm] = evaluate_prediction_set(
                    detector_predictions, coco_detector, detector_image_ids, strata,
                    detector_only=True,
                )
        report["resolutions"][size_text] = size_report

    # Controlled contrasts used for interpretation.
    for size_report in report["resolutions"].values():
        arms = size_report["arms"]
        size_report["contrasts"] = {}
        for label, right, left in [
            ("B_minus_A__P2_direct", "B_direct_p2", "A_direct_baseline"),
            ("C_minus_A__decoupling_baseline", "C_baseline_plus_efficientnet_b0", "A_direct_baseline"),
            ("D_minus_C__P2_two_stage", "D_p2_plus_efficientnet_b0", "C_baseline_plus_efficientnet_b0"),
            ("D_minus_B__decoupling_p2", "D_p2_plus_efficientnet_b0", "B_direct_p2"),
        ]:
            size_report["contrasts"][label] = {
                stratum: {
                    metric: arms[right][stratum][metric] - arms[left][stratum][metric]
                    for metric in ("AP50", "AP50_95")
                }
                for stratum in arms[right]
            }
        detectors = size_report["detector_only"]
        size_report["detector_contrast_D_minus_C"] = {
            stratum: {
                metric: detectors["D_p2_plus_efficientnet_b0"][stratum][metric]
                - detectors["C_baseline_plus_efficientnet_b0"][stratum][metric]
                for metric in ("AP50", "AP50_95")
            }
            for stratum in detectors["D_p2_plus_efficientnet_b0"]
        }

    target = OUTPUT / "scale_stratified_ap.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"[COMPLETE] {target}", flush=True)


if __name__ == "__main__":
    main()
