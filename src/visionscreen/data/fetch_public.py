"""Best-effort acquisition of openly licensed real-world reference images.

Searches Wikimedia Commons (openly licensed media) for clinical eye-condition
examples and downloads them with full provenance logging. This builds the
small-N reality-check set from the spec — a curation aid, not a training
pipeline. Every failure is logged and non-fatal.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "visionscreen-research/0.1 (open-source vision screening research)"

SEARCH_TERMS = [
    "strabismus eyes",
    "esotropia",
    "exotropia",
    "red reflex eye",
    "leukocoria",
    "Hirschberg test",
    "photorefraction",
]


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def parse_commons_response(data: dict, search_term: str) -> list[dict]:
    items = []
    for page in data.get("query", {}).get("pages", {}).values():
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        meta = info.get("extmetadata", {})
        items.append(
            {
                "title": page.get("title", ""),
                "url": info.get("thumburl") or info.get("url", ""),
                "license": meta.get("LicenseShortName", {}).get("value", "unknown"),
                "artist": meta.get("Artist", {}).get("value", "unknown"),
                "search_term": search_term,
                "source": "Wikimedia Commons",
            }
        )
    return items


def search_commons(term: str, limit: int = 4) -> list[dict]:
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {term}",
            "gsrnamespace": 6,
            "gsrlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "iiurlwidth": 1024,
            "format": "json",
        }
    )
    return parse_commons_response(_get_json(f"{API}?{params}"), term)


def fetch_all(out_dir: Path, terms: list[str] | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance: list[dict] = []
    errors: list[str] = []
    for term in terms or SEARCH_TERMS:
        try:
            items = search_commons(term)
        except Exception as exc:  # network failures are logged, never fatal
            errors.append(f"search '{term}': {exc}")
            continue
        for item in items:
            if not item["url"]:
                continue
            name = item["title"].replace("File:", "").replace("/", "_").replace(" ", "_")
            dest = out_dir / name
            if not dest.exists():
                time.sleep(2.0)  # pace per Wikimedia rate-limit guidance
                try:
                    req = urllib.request.Request(
                        item["url"], headers={"User-Agent": USER_AGENT}
                    )
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        dest.write_bytes(resp.read())
                except Exception as exc:
                    errors.append(f"download {item['title']}: {exc}")
                    continue
            item["local_path"] = str(dest)
            provenance.append(item)
    summary = {"items": provenance, "errors": errors, "n": len(provenance)}
    (out_dir / "provenance.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/real/commons")
    summary = fetch_all(out)
    print(f"fetched {summary['n']} items, {len(summary['errors'])} errors")
    for e in summary["errors"]:
        print("  !", e)


if __name__ == "__main__":
    main()
