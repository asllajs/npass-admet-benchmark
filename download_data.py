"""Fetch the Zenodo half of the input data into ``data/``.

    python download_data.py

Downloads the five files the analyses need from the open-data record
(10.5281/zenodo.22885507), about 170 MB in total. The 626 MB SQLite bundle and the
raw (untransformed) prediction table in the same record are convenience copies of
the same content and are not downloaded.

The Wang et al. (2016) drug-like Caco-2 benchmark is published as supporting
information by a third party and cannot be fetched automatically; see
``data/README.md``.
"""
from __future__ import annotations

import json
import shutil
import sys
import urllib.request

from npadmet import config as C

API = f"https://zenodo.org/api/records/{C.ZENODO_RECORD}"


def download(url: str, destination) -> None:
    with urllib.request.urlopen(url) as response, open(destination, "wb") as handle:
        shutil.copyfileobj(response, handle)


def main() -> int:
    C.DATA.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(API) as response:
        record = json.load(response)

    entries = {entry["key"]: entry for entry in record["files"]}
    missing = [name for name in C.ZENODO_FILES if name not in entries]
    if missing:
        print(f"record {C.ZENODO_RECORD} does not contain: {', '.join(missing)}")
        return 1

    for name in C.ZENODO_FILES:
        destination = C.DATA / name
        if destination.exists():
            print(f"  [have] {name}")
            continue
        size_mb = entries[name]["size"] / 1e6
        print(f"  [get ] {name} ({size_mb:,.1f} MB) ...", flush=True)
        download(entries[name]["links"]["self"], destination)
    print(f"\ndata is in {C.DATA}")
    print("Still needed for the applicability-domain analysis: "
          f"{C.FILE_WANG.name} (see data/README.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
