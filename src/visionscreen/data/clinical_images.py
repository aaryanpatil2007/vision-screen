"""Fetch clinically *categorised* images from Wikimedia Commons.

Keyword search returns cross-eyed owls and roses. Category membership is
curated by editors, so `Category:Esotropia` genuinely contains photographs of
people diagnosed with esotropia — which makes it usable as ground truth for a
small real-patient validation, in a way a text search never is.

Every item keeps its category (the label), URL, license and fetch date.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "visionscreen-research/0.3 (open vision screening research)"}

# category -> (label, is_positive_for_misalignment)
CATEGORIES = {
    "Strabismus": ("strabismus", True),
    "Esotropia": ("esotropia", True),
    "Exotropia": ("exotropia", True),
    "Hypertropia": ("hypertropia", True),
    "Strabismus_in_humans": ("strabismus", True),
    "Leukocoria": ("leukocoria", None),      # different condition, kept for reference
}

# Titles that are obviously not clinical human photographs. Category pages do
# contain diagrams, animals and unrelated media.
_REJECT_TOKENS = (
    "svg", "diagram", "chart", "logo", "icon", "toxin", "molecul",
    "dog", "cat", "owl", "tiger", "bulldog", "bouledogue", "animal",
    "autostereogram", "graph", "map",
    # historical engravings and anatomical plates: real strabismus, but drawings
    "traité", "traite_complet", "anatomie", "gravure", "engraving", "plate",
    "imbecile", "vache", "abrollstrecke", "oxycephalus", "1900", "18",
)


_LAST_CALL = [0.0]
API_MIN_INTERVAL_S = 1.5


def _get_json(url: str, retries: int = 4) -> dict:
    """Paced + retried: Wikimedia rate-limits API reads as well as downloads,
    and an unpaced sweep silently loses whole categories."""
    for attempt in range(retries):
        gap = API_MIN_INTERVAL_S - (time.monotonic() - _LAST_CALL[0])
        if gap > 0:
            time.sleep(gap)
        _LAST_CALL[0] = time.monotonic()
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def category_files(category: str, limit: int = 40) -> list[str]:
    params = urllib.parse.urlencode({
        "action": "query", "list": "categorymembers",
        "cmtitle": f"Category:{category}", "cmtype": "file",
        "cmlimit": limit, "format": "json",
    })
    data = _get_json(f"{API}?{params}")
    return [m["title"] for m in data.get("query", {}).get("categorymembers", [])]


def _plausibly_clinical(title: str) -> bool:
    low = title.lower()
    if not low.endswith((".jpg", ".jpeg", ".png", ".webp")):
        return False
    return not any(tok in low for tok in _REJECT_TOKENS)


def file_info(title: str) -> dict | None:
    params = urllib.parse.urlencode({
        "action": "query", "titles": title, "prop": "imageinfo",
        "iiprop": "url|extmetadata", "iiurlwidth": 1024, "format": "json",
    })
    pages = _get_json(f"{API}?{params}").get("query", {}).get("pages", {})
    for page in pages.values():
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        meta = info.get("extmetadata", {})
        return {
            "title": title,
            "url": info.get("thumburl") or info.get("url"),
            "license": meta.get("LicenseShortName", {}).get("value", "unknown"),
            "artist": meta.get("Artist", {}).get("value", "unknown"),
        }
    return None


def fetch(out_dir: Path, fetched_date: str, pause_s: float = 2.0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    items, errors = [], []
    for category, (label, positive) in CATEGORIES.items():
        try:
            titles = [t for t in category_files(category) if _plausibly_clinical(t)]
        except Exception as exc:
            errors.append(f"category {category}: {exc}")
            continue
        for title in titles:
            try:
                info = file_info(title)
            except Exception as exc:
                errors.append(f"info {title}: {exc}")
                continue
            if not info or not info["url"]:
                continue
            name = title.replace("File:", "").replace("/", "_").replace(" ", "_")
            dest = out_dir / name
            if not dest.exists():
                time.sleep(pause_s)   # Wikimedia rate-limit guidance
                try:
                    req = urllib.request.Request(info["url"], headers=UA)
                    with urllib.request.urlopen(req, timeout=60) as r:
                        dest.write_bytes(r.read())
                except Exception as exc:
                    errors.append(f"download {title}: {exc}")
                    continue
            items.append({
                **info, "category": category, "label": label,
                "misalignment_positive": positive,
                "local_path": str(dest), "fetched": fetched_date,
                "source": "Wikimedia Commons category membership",
            })
    summary = {"n": len(items), "items": items, "errors": errors}
    (out_dir / "labels.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real/clinical")
    ap.add_argument("--date", required=True)
    args = ap.parse_args()
    s = fetch(Path(args.out), args.date)
    print(f"fetched {s['n']} categorised clinical images, {len(s['errors'])} errors")
    by = {}
    for it in s["items"]:
        by.setdefault(it["label"], []).append(it["title"])
    for label, titles in by.items():
        print(f"  {label}: {len(titles)}")


if __name__ == "__main__":
    main()
