"""Acquire real eye imagery from openly downloadable Hugging Face datasets.

Everything here is fetchable without auth. Each dataset writes a
provenance.json recording the source repo, license, file list and fetch date,
so the corpus can be defended in the writeup.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HF_API = "https://huggingface.co/api/datasets"
HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
UA = {"User-Agent": "visionscreen-research/0.2"}


@dataclass(frozen=True)
class HFSource:
    repo: str
    description: str
    suffixes: tuple[str, ...] = (".jpg", ".jpeg", ".png")
    max_files: int = 400


SOURCES = (
    HFSource(
        repo="RafeiKAr/eye_tracking_gazecapture",
        description="GazeCapture-derived face frames (real webcam-style captures)",
        max_files=1200,
    ),
    HFSource(
        repo="hungryfull/Pupil_Position_in_the_Eye",
        description="Close-up eye crops with pupil in varied gaze positions",
        max_files=200,
    ),
)


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def list_files(repo: str, suffixes: tuple[str, ...]) -> list[str]:
    info = _get_json(f"{HF_API}/{repo}")
    return [
        s["rfilename"]
        for s in info.get("siblings", [])
        if s["rfilename"].lower().endswith(suffixes)
    ]


def dataset_license(repo: str) -> str:
    info = _get_json(f"{HF_API}/{repo}")
    return (info.get("cardData") or {}).get("license") or "unspecified"


def fetch_source(src: HFSource, out_root: Path, fetched_date: str) -> dict:
    out_dir = out_root / src.repo.replace("/", "__")
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        files = list_files(src.repo, src.suffixes)
        lic = dataset_license(src.repo)
    except Exception as exc:
        return {"repo": src.repo, "error": str(exc), "n": 0}

    # evenly sample across the listing so we span subjects, not one burst
    if len(files) > src.max_files:
        step = len(files) / src.max_files
        files = [files[int(i * step)] for i in range(src.max_files)]

    got, errors = [], []
    for rel in files:
        dest = raw_dir / rel.replace("/", "__")
        if dest.exists():
            got.append(rel)
            continue
        url = HF_RESOLVE.format(repo=src.repo, path=urllib.parse.quote(rel))
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                dest.write_bytes(r.read())
            got.append(rel)
        except Exception as exc:
            errors.append(f"{rel}: {exc}")

    prov = {
        "repo": src.repo,
        "source": f"https://huggingface.co/datasets/{src.repo}",
        "description": src.description,
        "license": lic,
        "fetched": fetched_date,
        "n_downloaded": len(got),
        "n_errors": len(errors),
        "errors": errors[:20],
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2))
    return prov


def fetch_all(out_root: Path, fetched_date: str) -> list[dict]:
    out_root.mkdir(parents=True, exist_ok=True)
    return [fetch_source(s, out_root, fetched_date) for s in SOURCES]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real/hf")
    ap.add_argument("--date", required=True, help="fetch date, e.g. 2026-08-04")
    args = ap.parse_args()
    for prov in fetch_all(Path(args.out), args.date):
        print(prov.get("repo"), "->", prov.get("n_downloaded", 0), "files",
              f"({prov.get('n_errors', 0)} errors)")


if __name__ == "__main__":
    main()
