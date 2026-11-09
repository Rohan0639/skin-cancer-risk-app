"""Download a real, labelled lesion dataset from the ISIC Archive public API.

This is what makes lesion-*type* classification possible. The synthetic demo
set has no dermatological labels, so a model trained on it can only ever
produce risk tiers; naming a diagnosis from it would be fabrication.

Images are restricted to ``clinical: close-up`` — photographs taken with an
ordinary camera, which is what this app receives. The archive is mostly
dermoscopic (taken through a contact lens against the skin), and training on
those would mean evaluating on a domain the app never sees. Mixing the two
would be worse still: the classes are unevenly represented across image types,
so the model could score well by learning "dermoscopic therefore melanoma".

Six classes, matching PAD-UFES-20's scheme and the existing risk mapping in
``configs/default.yaml``.

Usage::

    python data/prepare_isic.py --per-class 400
    python data/prepare_isic.py --per-class 800 --workers 16
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlencode

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, load_config  # noqa: E402
from src.data_prep import rebuild_unified, relative_to_repo, subsample, summarise  # noqa: E402
from src.dataset import UNIFIED_COLUMNS, normalise_age, write_unified_csv  # noqa: E402
from src.utils import ensure_dir, get_logger  # noqa: E402

logger = get_logger("prepare_isic")

SOURCE_NAME = "isic"
API = "https://api.isic-archive.com/api/v2/images/search/"

#: Project class -> the ISIC query selecting it. Melanoma is taken at the
#: diagnosis_2 level so in-situ and invasive subtypes are all included;
#: without that there are too few clinical melanomas to train on.
CLASS_QUERIES: Dict[str, str] = {
    "MEL": 'diagnosis_2:"Malignant melanocytic proliferations (Melanoma)"',
    "NEV": 'diagnosis_3:"Nevus"',
    "BCC": 'diagnosis_3:"Basal cell carcinoma"',
    "ACK": 'diagnosis_3:"Solar or actinic keratosis"',
    "SEK": 'diagnosis_3:"Seborrheic keratosis"',
    "SCC": 'diagnosis_3:"Squamous cell carcinoma, NOS"',
}

IMAGE_TYPE = 'image_type:"clinical: close-up"'

#: ISIC anatom_site_1 -> the project's canonical site vocabulary.
SITE_MAP = {
    "head and neck": "head_neck",
    "trunk": "trunk",
    "upper extremity": "upper_extremity",
    "lower extremity": "lower_extremity",
    "palms and soles": "palms_soles",
    "anogenital region": "genital",
}


def fetch_page(query: str, limit: int, cursor: Optional[str]) -> dict:
    """Fetch one page of search results, following the API's cursor."""
    import urllib.request

    url = cursor or f"{API}?{urlencode({'query': query, 'limit': limit})}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        import json

        return json.loads(resp.read().decode("utf-8"))


def collect(label: str, want: int, page_size: int = 100) -> List[dict]:
    """Page through the API collecting up to ``want`` records for one class."""
    query = f"{CLASS_QUERIES[label]} AND {IMAGE_TYPE}"
    out: List[dict] = []
    cursor = None
    while len(out) < want:
        try:
            page = fetch_page(query, min(page_size, want - len(out)), cursor)
        except Exception as exc:  # noqa: BLE001
            logger.warning("  %s: page fetch failed (%s); stopping early.", label, exc)
            break
        results = page.get("results") or []
        if not results:
            break
        out.extend(results)
        cursor = page.get("next")
        if not cursor:
            break
    logger.info("  %-4s %4d records", label, len(out))
    return out[:want]


def download_one(record: dict, dest_dir: Path) -> Optional[Path]:
    """Download one image's 256px thumbnail. Returns the path, or None."""
    import urllib.request

    isic_id = record.get("isic_id")
    files = record.get("files") or {}
    url = (files.get("thumbnail_256") or {}).get("url")
    if not isic_id or not url:
        return None
    path = dest_dir / f"{isic_id}.jpg"
    if path.exists() and path.stat().st_size > 0:
        return path
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "skin-cancer-risk-app"})
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = resp.read()
        if len(data) < 512:
            return None
        path.write_bytes(data)
        return path
    except Exception:  # noqa: BLE001
        return None


def prepare(cfg: Config, per_class: int, workers: int, merge: bool = True) -> Optional[Path]:
    """Download the subset and write ``data/processed/isic.csv``."""
    raw_root = ensure_dir(cfg.path("paths.raw_dir") / SOURCE_NAME)
    risk_map = {k.upper(): v for k, v in dict(cfg.get("labels.pad_ufes_map")).items()}
    site_vocab = list(cfg.get("metadata.site_vocab"))

    logger.info("Querying the ISIC Archive for clinical close-up images ...")
    records: Dict[str, List[dict]] = {}
    for label in CLASS_QUERIES:
        records[label] = collect(label, per_class)

    rows = []
    started = time.time()
    for label, recs in records.items():
        if not recs:
            logger.warning("No records for %s -- skipping.", label)
            continue
        dest = ensure_dir(raw_root / label)
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(download_one, r, dest): r for r in recs}
            for fut in as_completed(futures):
                path = fut.result()
                rec = futures[fut]
                done += 1
                if path is None:
                    continue
                clinical = (rec.get("metadata") or {}).get("clinical") or {}
                site_raw = str(clinical.get("anatom_site_1") or "").strip().lower()
                rows.append({
                    "image_path": relative_to_repo(path),
                    "age": normalise_age(clinical.get("age_approx"), cfg),
                    "sex": str(clinical.get("sex") or "unknown").strip().lower(),
                    "site": SITE_MAP.get(site_raw, "unknown") if site_raw else "unknown",
                    "diagnosis": label,
                    "risk_label": risk_map.get(label, "Moderate"),
                    "source": SOURCE_NAME,
                    # The archive groups repeat shots of one lesion under
                    # lesion_id; fall back to the image id so every row has a
                    # group key for the leakage-free split.
                    "lesion_id": str(clinical.get("lesion_id")
                                     or rec.get("isic_id") or f"{SOURCE_NAME}_{done}"),
                })
        logger.info("  %-4s downloaded %d/%d (%.0fs elapsed)",
                    label, len([r for r in rows if r["diagnosis"] == label]),
                    len(recs), time.time() - started)

    if not rows:
        logger.error("No images were downloaded.")
        return None

    df = pd.DataFrame(rows, columns=UNIFIED_COLUMNS)
    df["lesion_id"] = SOURCE_NAME + "_" + df["lesion_id"].astype(str)
    df = subsample(df, None, None, int(cfg.get("project.seed", 42)))

    logger.info("--- ISIC: %d images ---", len(df))
    for dx, count in df["diagnosis"].value_counts().items():
        logger.info("  %-4s %5d", dx, count)
    summarise(df, "ISIC (real clinical photographs)")

    out_path = cfg.path("paths.processed_dir") / f"{SOURCE_NAME}.csv"
    write_unified_csv(df, out_path)
    if merge:
        rebuild_unified(cfg)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--per-class", type=int, default=400,
                        help="Max images per diagnosis class (default 400).")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--no-merge", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = prepare(cfg, args.per_class, args.workers, merge=not args.no_merge)
    return 0 if result is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
