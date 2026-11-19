"""Extract sharp, de-duplicated frames from aquarium video clips.

Usage:
    python process_videos.py --videos raw/videos --out frames --manifest manifest.csv

The immediate parent folder of each video is treated as its "tank".  Frames keep
the source clip stem in their filename so that later train/val/test splits can be
made per tank/clip instead of per frame (avoiding leakage).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def list_videos(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def process_video(video: Path, args: argparse.Namespace) -> list[dict]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    sample_step = max(1, round(fps / args.fps))

    sampled = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % sample_step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sampled.append((index, index / fps, sharpness(gray)))
        index += 1
    cap.release()

    if not sampled:
        return []

    max_sharp = max(s for _, _, s in sampled)
    floor = max(args.min_sharpness, args.min_sharpness_ratio * max_sharp)

    eligible = [(idx, sec, s) for idx, sec, s in sampled if s >= floor]
    eligible.sort(key=lambda item: item[2], reverse=True)

    min_gap_frames = max(1.0, args.min_gap * fps)
    chosen: list[int] = []
    for idx, sec, s in eligible:
        if all(abs(idx - kept) >= min_gap_frames for kept in chosen):
            chosen.append(idx)
        if len(chosen) >= args.max_frames:
            break
    chosen = set(chosen)
    sharp_lookup = {idx: s for idx, sec, s in sampled}

    cap = cv2.VideoCapture(str(video))
    records = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index in chosen:
            tank = video.parent.name
            out_dir = args.out / tank
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / f"{video.stem}_{index:06d}.jpg"
            cv2.imwrite(str(target), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            sharp = sharp_lookup[index]
            records.append(
                {
                    "tank": tank,
                    "video": video.name,
                    "frame": target.name,
                    "seconds": round(index / fps, 2),
                    "sharpness": round(sharp, 3),
                    "path": str(target),
                }
            )
        index += 1
    cap.release()
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, default=Path("raw/videos"))
    parser.add_argument("--out", type=Path, default=Path("frames"))
    parser.add_argument("--manifest", type=Path, default=Path("manifest.csv"))
    parser.add_argument("--fps", type=float, default=0.5, help="sampling rate in frames/second")
    parser.add_argument("--min-gap", type=float, default=1.0, help="minimum seconds between kept frames")
    parser.add_argument("--max-frames", type=int, default=12, help="max frames kept per clip")
    parser.add_argument("--min-sharpness-ratio", type=float, default=0.3, help="keep frames whose sharpness is at least this fraction of the clip maximum")
    parser.add_argument("--min-sharpness", type=float, default=0.0, help="optional absolute sharpness floor")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    videos = list_videos(args.videos)
    if not videos:
        raise SystemExit(f"no videos found under {args.videos}")

    rows = []
    for video in videos:
        records = process_video(video, args)
        rows.extend(records)
        print(f"{video} -> {len(records)} frames", flush=True)

    with args.manifest.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tank", "video", "frame", "seconds", "sharpness", "path"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"total: {len(videos)} videos, {len(rows)} frames -> {args.manifest}")


if __name__ == "__main__":
    main()
