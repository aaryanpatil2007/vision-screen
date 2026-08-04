"""Turn downloaded real photos into a weakly-labeled eye-crop training corpus.

For every image: detect the face, take both eye crops via the same code path
the analyzer uses, and weakly label them from the iris prior. Rejected crops
are counted, not silently dropped — the acceptance rate is a reported number.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from visionscreen.data.weak_labels import weak_label_eye_crop
from visionscreen.perception.iris import eye_crop, iris_center, iris_diameter_px
from visionscreen.perception.landmarks import LandmarkExtractor


def build_corpus(
    raw_dirs: list[Path],
    out_root: Path,
    limit: int | None = None,
    min_crop_px: int = 26,
) -> dict:
    img_dir = out_root / "images"
    mask_dir = out_root / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for d in raw_dirs:
        if d.is_dir():
            paths += sorted(
                p for p in d.rglob("*")
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
    if limit:
        paths = paths[:limit]

    stats = {"images": 0, "faces": 0, "crops": 0, "labeled": 0, "rejected": 0}
    with LandmarkExtractor() as extractor:
        for p in paths:
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            stats["images"] += 1
            face = extractor.extract(frame)
            if not face.ok:
                continue
            stats["faces"] += 1
            h, w = frame.shape[:2]
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for side in ("left", "right"):
                crop_bgr, (ox, oy) = eye_crop(frame, face.landmarks, side, pad=0.55)
                if crop_bgr.size == 0 or min(crop_bgr.shape[:2]) < min_crop_px:
                    continue
                stats["crops"] += 1
                crop = gray_full[oy:oy + crop_bgr.shape[0], ox:ox + crop_bgr.shape[1]]
                c = iris_center(face.landmarks, side) * (w, h)
                radius = iris_diameter_px(face.landmarks, side, w, h) / 2
                mask = weak_label_eye_crop(crop, (c[0] - ox, c[1] - oy), radius)
                if mask is None:
                    stats["rejected"] += 1
                    continue
                name = f"{p.stem}_{side}.png"
                cv2.imwrite(str(img_dir / name), crop)
                cv2.imwrite(str(mask_dir / name), mask)
                stats["labeled"] += 1

    stats["acceptance_rate"] = (
        round(stats["labeled"] / stats["crops"], 3) if stats["crops"] else 0.0
    )
    (out_root / "corpus_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", default=["data/real/hf", "data/real/commons"])
    ap.add_argument("--out", default="data/corpus/real")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    stats = build_corpus([Path(r) for r in args.raw], Path(args.out), args.limit)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
