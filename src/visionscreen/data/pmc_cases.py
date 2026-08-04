"""Mine open-access case reports for photograph + clinician-measured deviation.

This is the closest thing to a clinical validation set that can be assembled
without recruiting patients. Published strabismus case reports routinely show a
patient photograph and state the prism cover test result — measured by a
clinician — in the same figure caption. That is genuinely paired data: a real
patient, a real doctor's measurement, an image.

The caveats are real and are carried through to the results: the caption's
deviation may refer to a different gaze position or a different time point than
the photograph (pre- vs post-operative figures are common), camera distance and
magnification are unknown, and captions are written for humans rather than
parsers. Every extracted pair therefore keeps its caption so a claim can be
checked, and the parser is deliberately conservative — it would rather miss a
pair than invent one.
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
FULLTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
UA = {"User-Agent": "visionscreen-research/0.4 (open vision screening research)"}

QUERY = (
    '(esotropia OR exotropia OR strabismus) AND "prism diopters" '
    'AND OPEN_ACCESS:y AND HAS_FT:y AND IN_EPMC:y'
)

# "35 prism diopters", "35 PD", "35Δ", "35 prism dioptres"
_DEV = re.compile(
    r"(\d{1,3})\s*(?:∆|Δ|PD\b|prism\s+di[oe]pt(?:er|re)s?)", re.IGNORECASE
)
# captions that plausibly show a face rather than a chart or an intraoperative view
_PHOTO_HINTS = ("photograph", "photo", "appearance", "preoperative", "pre-operative",
                "postoperative", "post-operative", "patient", "external", "clinical")
# Matched as whole words. Substring matching silently discarded every caption
# containing "photograph" (it contains "graph") and "octopus"/"doctor" style
# words for "oct" — i.e. exactly the figures this tool exists to find.
_REJECT_WORDS = (
    "schematic", "diagram", "chart", "graph", "flowchart", "intraoperative",
    "oct", "fundus", "mri", "histology", "histological", "b-scan",
    "pmc-status", "pmc-release", "responsedate",
)
_REJECT_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _REJECT_WORDS) + r")\b", re.IGNORECASE
)


def _get(url: str, retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def search(page_size: int = 100, pages: int = 4) -> list[str]:
    ids, cursor = [], "*"
    for _ in range(pages):
        params = urllib.parse.urlencode({
            "query": QUERY, "format": "json", "pageSize": page_size,
            "cursorMark": cursor, "resultType": "lite",
        })
        data = json.loads(_get(f"{SEARCH}?{params}").decode())
        for r in data.get("resultList", {}).get("result", []):
            if r.get("pmcid"):
                ids.append(r["pmcid"])
        nxt = data.get("nextCursorMark")
        if not nxt or nxt == cursor:
            break
        cursor = nxt
        time.sleep(0.5)
    return ids


def parse_figures(xml: str, pmcid: str) -> list[dict]:
    """Pull (figure graphic, caption, stated deviation) triples from JATS XML."""
    out = []
    for m in re.finditer(r"<fig\b.*?</fig>", xml, re.DOTALL):
        block = m.group(0)
        graphic = re.search(r'xlink:href="([^"]+)"', block)
        if not graphic:
            continue
        # caption text only — <fig> siblings can carry processing metadata
        cap_m = re.search(r"<caption\b.*?</caption>", block, re.DOTALL)
        caption = re.sub(r"<[^>]+>", " ", cap_m.group(0) if cap_m else block)
        caption = re.sub(r"\s+", " ", caption).strip()
        low = caption.lower()
        if _REJECT_RE.search(low):
            continue
        if not any(hint in low for hint in _PHOTO_HINTS):
            continue
        devs = [int(d) for d in _DEV.findall(caption)]
        devs = [d for d in devs if 1 <= d <= 120]
        if len(devs) != 1:
            # zero is useless; more than one is ambiguous about which eye or
            # which time point the photograph shows
            continue
        out.append({
            "pmcid": pmcid,
            "graphic": graphic.group(1),
            "caption": caption[:400],
            "deviation_pd": devs[0],
        })
    return out


def fetch(out_dir: Path, fetched_date: str, max_articles: int = 400) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pmcids = search()[:max_articles]
    items, errors = [], []
    for pmcid in pmcids:
        try:
            xml = _get(FULLTEXT.format(pmcid=pmcid)).decode("utf-8", "replace")
        except Exception as exc:
            errors.append(f"{pmcid}: {exc}")
            continue
        for fig in parse_figures(xml, pmcid):
            # NOTE: figure images are NOT retrievable from every network.
            # NCBI and Europe PMC both restrict direct image endpoints by
            # origin, so this tool records the canonical locations and leaves
            # retrieval to an environment that has access. The OA package is
            # the supported bulk route.
            fig["image_candidates"] = [
                f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/bin/{fig['graphic']}",
                f"https://europepmc.org/articles/{pmcid}/bin/{fig['graphic']}",
            ]
            fig["oa_package_lookup"] = (
                f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
            )
            fig["source"] = f"https://europepmc.org/article/PMC/{pmcid}"
            fig["fetched"] = fetched_date
            items.append(fig)
        time.sleep(0.34)

    summary = {"n": len(items), "articles_scanned": len(pmcids),
               "items": items, "errors": errors[:20]}
    (out_dir / "cases.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real/pmc_cases")
    ap.add_argument("--date", required=True)
    ap.add_argument("--max-articles", type=int, default=400)
    args = ap.parse_args()
    s = fetch(Path(args.out), args.date, args.max_articles)
    print(f"scanned {s['articles_scanned']} articles, "
          f"found {s['n']} figure/deviation pairs, {len(s['errors'])} errors")
    for it in s["items"][:10]:
        print(f"  {it['pmcid']:12s} {it['deviation_pd']:3d} PD  {it['caption'][:70]}")


if __name__ == "__main__":
    main()
