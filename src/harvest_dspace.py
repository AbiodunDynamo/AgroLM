#!/usr/bin/env python3
"""
harvest_dspace.py

Resumable harvester for DSpace-based repositories (CGSpace, FAO Knowledge
Repository) for the tinyLM-Agri / AgroLM pretraining corpus.

Design (agreed):
- Broad per-crop queries only (no crop x topic pairing). 5 crops x N sources.
- Resumable via a per-crop offset stored in state.json. manifest.jsonl is the
  source of truth for dedup (seen UUIDs) so re-running never re-saves a doc.
- Each run stops after fetching MAX_NEW_DOCS_PER_RUN new documents, or once
  every crop query is exhausted for every source -- whichever comes first.
- Cheap metadata (title + abstract) is checked against an exclusion keyword
  list BEFORE the full text is downloaded. Excluded items are logged to
  excluded.jsonl and never touch data/raw/.
- Passing items have their full text extracted from the primary bitstream
  (PDF) and saved to data/raw/<source>/<uuid>.txt. A row is appended to
  manifest.jsonl.

Usage (from repo root, e.g. in Colab after cloning AgroLM):
    python src/harvest_dspace.py --source cgspace
    python src/harvest_dspace.py --source fao

Run repeatedly (once per Colab session, or multiple times per session) to
keep progressing through each crop's result set. State and manifest live in
data/raw/ and are committed back to GitHub at the end of each session --
this script does not push to git itself.
"""

import argparse
import hashlib
import io
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

SOURCES = {
    "cgspace": {
        "base_url": "https://cgspace.cgiar.org/server/api",
    },
    "fao": {
        "base_url": "https://openknowledge.fao.org/server/api",
    },
}

CROPS = ["maize", "cassava", "tomato", "rice", "cowpea"]

# Topic keywords used ONLY for post-hoc tagging in the manifest, never to
# constrain the query itself.
TOPIC_KEYWORDS = {
    "pest_disease": ["pest", "disease", "insect", "fungus", "fungal", "blight", "weevil", "aphid"],
    "planting_timing": ["planting", "sowing", "seeding", "calendar", "season"],
    "soil_fertilizer": ["soil", "fertilizer", "fertiliser", "nutrient", "manure", "compost"],
    "irrigation_water": ["irrigation", "water management", "drought", "rainfall"],
    "climate_adaptation": ["climate change", "climate adaptation", "resilience", "climate-smart"],
}

# Title/abstract exclusion list.
# Category 1: molecular/genomic research methodology -- not breeding outcomes
# ("drought-tolerant maize variety released") which should stay.
# Category 2: CGIAR institutional/administrative documents -- funding
# proposals, program evaluations, strategy docs. These rank highly on bare
# crop queries against CGSpace since CGIAR is the org running it, but they
# are not advisory content and tend to be very long, skewing corpus token
# counts without adding real diversity. Checked on title alone since these
# titles are unambiguous.
EXCLUSION_KEYWORDS = [
    # genomic/molecular methodology
    "genome", "genomic", "sequencing", "qtl", "marker-assisted",
    "transcriptome", "proteomics", "phylogenetic", "allele",
    "genotyping", "molecular marker", "snp",
    # institutional/administrative
    "research program", "full proposal", "evaluation of", "call for proposals",
    "strategic plan", "annual report", "project proposal", "concept note",
    "terms of reference", "workshop report", "meeting report",
]

MAX_NEW_DOCS_PER_RUN = 100
MAX_DOCS_PER_CROP_PER_RUN = -(-MAX_NEW_DOCS_PER_RUN // len(CROPS))  # ceil division, e.g. 20 for 5 crops
PAGE_SIZE = 20  # DSpace discover/search default-friendly page size
REQUEST_DELAY_SECONDS = 1.5  # politeness / rate limiting between requests
REQUEST_TIMEOUT = 30

DATA_ROOT = Path("data/raw")
MANIFEST_PATH = DATA_ROOT / "manifest.jsonl"
EXCLUDED_PATH = DATA_ROOT / "excluded.jsonl"
STATE_PATH = DATA_ROOT / "state.json"

USER_AGENT = "AgroLM-corpus-harvester/0.1 (research project; contact via GitHub repo)"


# --------------------------------------------------------------------------
# State + manifest helpers
# --------------------------------------------------------------------------

def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def get_query_state(state, source, crop):
    key = f"{source}:{crop}"
    return state.setdefault(key, {"offset": 0, "exhausted": False})


def load_seen_uuids():
    seen = set()
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                seen.add(row["uuid"])
    return seen


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# Filtering / tagging
# --------------------------------------------------------------------------

def is_excluded(title, abstract):
    text = f"{title or ''} {abstract or ''}".lower()
    for kw in EXCLUSION_KEYWORDS:
        if kw in text:
            return kw
    return None


def tag_topics(title, abstract):
    text = f"{title or ''} {abstract or ''}".lower()
    tags = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            tags.append(topic)
    return tags


def guess_register(title, abstract):
    text = f"{title or ''} {abstract or ''}".lower()
    if any(w in text for w in ["bulletin", "extension", "guide", "advisory", "fact sheet", "factsheet"]):
        return "advisory-bulletin"
    if any(w in text for w in ["abstract", "journal", "study", "research", "trial"]):
        return "research-abstract"
    if any(w in text for w in ["brief", "policy brief", "practice brief"]):
        return "practice-brief"
    return "other"


# --------------------------------------------------------------------------
# DSpace API interaction
# --------------------------------------------------------------------------

def api_get(url, params=None):
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)
    return resp.json()


def search_items(base_url, query, page, size):
    """
    DSpace 7.x discover/search/objects endpoint.
    Returns list of raw item dicts (indexableObject) and whether more pages remain.
    """
    url = f"{base_url}/discover/search/objects"
    params = {
        "query": query,
        "dsoType": "item",
        "page": page,
        "size": size,
    }
    data = api_get(url, params=params)

    try:
        objects = data["_embedded"]["searchResult"]["_embedded"]["objects"]
    except KeyError:
        return [], False

    items = []
    for obj in objects:
        indexable = obj.get("_embedded", {}).get("indexableObject")
        if indexable:
            items.append(indexable)

    page_info = data.get("_embedded", {}).get("searchResult", {}).get("page", {})
    total_pages = page_info.get("totalPages", 1)
    has_more = (page + 1) < total_pages

    return items, has_more


def extract_metadata_fields(item):
    md = item.get("metadata", {})

    def first(field):
        vals = md.get(field)
        if vals:
            return vals[0].get("value")
        return None

    return {
        "uuid": item.get("uuid"),
        "title": first("dc.title"),
        "abstract": first("dc.description.abstract"),
        "date_issued": first("dc.date.issued"),
        "rights": first("dc.rights") or first("dc.rights.license"),
    }


def get_bitstreams(base_url, item_uuid):
    """
    Walk item -> bundles -> bitstreams to find the primary content file
    (typically the ORIGINAL bundle's first bitstream, usually a PDF).
    """
    url = f"{base_url}/core/items/{item_uuid}/bundles"
    data = api_get(url)
    bundles = data.get("_embedded", {}).get("bundles", [])

    for bundle in bundles:
        if bundle.get("name") != "ORIGINAL":
            continue
        bundle_id = bundle.get("uuid")
        bitstreams_url = f"{base_url}/core/bundles/{bundle_id}/bitstreams"
        bdata = api_get(bitstreams_url)
        bitstreams = bdata.get("_embedded", {}).get("bitstreams", [])
        return bitstreams

    return []


def download_bitstream_text(base_url, bitstream):
    """
    Download a bitstream and extract text. Only handles PDF and plain text;
    anything else is skipped (returns None).
    """
    content_url = f"{base_url}/core/bitstreams/{bitstream['uuid']}/content"
    mime_type = bitstream.get("format", {}).get("mimetype", "") if isinstance(bitstream.get("format"), dict) else ""

    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(content_url, headers=headers, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)

    name = bitstream.get("name", "")
    if name.lower().endswith(".pdf") or "pdf" in mime_type.lower():
        return extract_pdf_text(resp.content)
    elif name.lower().endswith(".txt") or "text/plain" in mime_type.lower():
        return resp.content.decode("utf-8", errors="ignore")
    else:
        return None


def extract_pdf_text(pdf_bytes):
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError(
            "pypdf is required to extract PDF text. Install with: "
            "pip install pypdf --break-system-packages"
        )
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(pages_text).strip()


# --------------------------------------------------------------------------
# Main harvest loop
# --------------------------------------------------------------------------

def harvest(source):
    base_url = SOURCES[source]["base_url"]
    state = load_state()
    seen_uuids = load_seen_uuids()

    new_docs_saved = 0
    per_crop_saved = {crop: 0 for crop in CROPS}
    # Queue-based round robin: a crop is re-appended to the back of the queue
    # after each page as long as it isn't exhausted and hasn't hit its
    # per-run cap yet. This guarantees every crop gets touched each run
    # instead of one crop draining the entire MAX_NEW_DOCS_PER_RUN budget.
    crops_queue = list(CROPS)

    print(f"[{source}] starting run. Already seen {len(seen_uuids)} docs across all sources. "
          f"Per-crop cap this run: {MAX_DOCS_PER_CROP_PER_RUN}")

    while crops_queue and new_docs_saved < MAX_NEW_DOCS_PER_RUN:
        crop = crops_queue.pop(0)
        qstate = get_query_state(state, source, crop)

        if qstate["exhausted"]:
            continue  # permanently done for this crop/source, don't re-queue

        if per_crop_saved[crop] >= MAX_DOCS_PER_CROP_PER_RUN:
            continue  # hit this run's cap for this crop, don't re-queue this run

        page = qstate["offset"] // PAGE_SIZE
        print(f"[{source}:{crop}] fetching page {page} (offset {qstate['offset']})")

        try:
            items, has_more = search_items(base_url, crop, page, PAGE_SIZE)
        except requests.HTTPError as e:
            print(f"[{source}:{crop}] HTTP error: {e}. Skipping for this run, will retry next run.")
            continue

        if not items:
            qstate["exhausted"] = True
            save_state(state)
            continue

        for item in items:
            if new_docs_saved >= MAX_NEW_DOCS_PER_RUN:
                break
            if per_crop_saved[crop] >= MAX_DOCS_PER_CROP_PER_RUN:
                break

            meta = extract_metadata_fields(item)
            uuid = meta["uuid"]

            if not uuid or uuid in seen_uuids:
                continue

            excluded_kw = is_excluded(meta["title"], meta["abstract"])
            if excluded_kw:
                append_jsonl(EXCLUDED_PATH, {
                    "uuid": uuid,
                    "title": meta["title"],
                    "matched_keyword": excluded_kw,
                    "source_repo": source,
                    "crop_query": crop,
                    "excluded_at": datetime.now(timezone.utc).isoformat(),
                })
                seen_uuids.add(uuid)  # don't re-evaluate on future runs
                continue

            try:
                bitstreams = get_bitstreams(base_url, uuid)
            except requests.HTTPError as e:
                print(f"[{source}:{crop}] failed to list bitstreams for {uuid}: {e}")
                continue

            full_text = None
            for bitstream in bitstreams:
                try:
                    full_text = download_bitstream_text(base_url, bitstream)
                except Exception as e:
                    print(f"[{source}:{crop}] failed to extract text for {uuid}: {e}")
                    continue
                if full_text:
                    break

            if not full_text or len(full_text.strip()) < 200:
                # Nothing usable extracted (scanned image PDF, empty bundle, etc.)
                append_jsonl(EXCLUDED_PATH, {
                    "uuid": uuid,
                    "title": meta["title"],
                    "matched_keyword": "NO_EXTRACTABLE_TEXT",
                    "source_repo": source,
                    "crop_query": crop,
                    "excluded_at": datetime.now(timezone.utc).isoformat(),
                })
                seen_uuids.add(uuid)
                continue

            out_dir = DATA_ROOT / source
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{uuid}.txt"
            out_path.write_text(full_text, encoding="utf-8")

            content_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
            word_count = len(full_text.split())

            manifest_row = {
                "uuid": uuid,
                "source_repo": source,
                "content_hash": content_hash,
                "title": meta["title"],
                "date_issued": meta["date_issued"],
                "crop_tags": [crop],
                "topic_tags": tag_topics(meta["title"], meta["abstract"]),
                "register": guess_register(meta["title"], meta["abstract"]),
                "license_rights": meta["rights"],
                "word_count": word_count,
                "file_path": str(out_path),
                "harvested_at": datetime.now(timezone.utc).isoformat(),
            }
            append_jsonl(MANIFEST_PATH, manifest_row)

            seen_uuids.add(uuid)
            new_docs_saved += 1
            per_crop_saved[crop] += 1
            print(f"[{source}:{crop}] saved {uuid} ({word_count} words) "
                  f"[{per_crop_saved[crop]}/{MAX_DOCS_PER_CROP_PER_RUN} this crop, "
                  f"{new_docs_saved}/{MAX_NEW_DOCS_PER_RUN} this run]")

        qstate["offset"] += PAGE_SIZE
        if not has_more:
            qstate["exhausted"] = True
        save_state(state)

        if not qstate["exhausted"] and per_crop_saved[crop] < MAX_DOCS_PER_CROP_PER_RUN:
            crops_queue.append(crop)  # still has room and more pages -- rotate back for another turn

    print(f"[{source}] run complete. {new_docs_saved} new docs saved this run.")
    print(f"[{source}] per-crop breakdown: {per_crop_saved}")
    if not crops_queue:
        print(f"[{source}] all crop queries exhausted or capped for this run.")


def main():
    parser = argparse.ArgumentParser(description="Harvest AgroLM pretraining corpus from a DSpace repository.")
    parser.add_argument("--source", required=True, choices=list(SOURCES.keys()))
    args = parser.parse_args()
    harvest(args.source)


if __name__ == "__main__":
    main()
