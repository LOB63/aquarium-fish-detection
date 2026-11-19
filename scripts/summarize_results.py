"""Print and cross-check archived results without installing ML dependencies."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def main():
    reports = {}
    for size in (1280, 640):
        report = json.loads((ROOT / "results" / f"{size}_test_metrics.json").read_text(encoding="utf-8"))
        if report["test_images"] != 95 or report["test_instances"] != 3588:
            raise ValueError("Unexpected final-test protocol")
        reports[size] = report
        for arm, metrics in report["arms"].items():
            print(f"{size} | {arm} | mAP50={metrics['mAP50']:.4f} | mAP50_95={metrics['mAP50_95']:.4f}")
    report = reports[1280]["arms"]
    delta = report["p2_plus_efficientnet_b0"]["mAP50"] - report["A_direct_baseline"]["mAP50"]
    print(f"1280 D-A: {delta*100:.2f} percentage points")
    if round(delta*100,2) != 10.03:
        raise ValueError("Resume metric does not match archived JSON")
    scale = json.loads((ROOT / "results" / "scale_stratified_ap.json").read_text(encoding="utf-8"))
    if sum(scale["test_stratum_counts"].values()) != 3588:
        raise ValueError("Scale counts do not match the held-out set")

if __name__ == "__main__":
    main()
