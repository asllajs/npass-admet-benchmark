"""Data loading.

Two sources, deliberately kept separate so that a partial download still runs
whatever it can:

1. Zenodo (10.5281/zenodo.22885507) - everything this study produced. The
   experimental Caco-2 benchmark and the three tools' predictions for it, the
   201,905-compound harmonized prediction table, the biosynthetic classification
   and the endpoint crosswalk. ``python download_data.py`` fetches them.
2. Wang et al. (2016) supporting information - the drug-like Caco-2 reference set
   used for the applicability-domain contrast and the in-domain positive control,
   which belongs to a third party and has to be downloaded by hand.

Anything missing raises :class:`DataUnavailable`, which the test suite turns into
an explicit skip rather than a silent pass.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C


class DataUnavailable(RuntimeError):
    """Raised when an input file has not been downloaded."""


def _require(path, hint: str):
    if not path.exists():
        raise DataUnavailable(f"missing {path.name} in {path.parent} - {hint}")
    return path


_ZENODO_HINT = (f"run `python download_data.py`, or fetch it from Zenodo "
                f"({C.ZENODO_DOI}) into the data/ folder")
_WANG_HINT = (f"download the supporting-information spreadsheet of Wang et al. (2016), "
              f"{C.WANG_URL}, and save it as data/{C.FILE_WANG.name}")
_KIM_HINT = (f"download the supporting-information document of Kim et al. (2026), "
             f"{C.KIM_URL}, and save it as data/{C.FILE_KIM.name}")


# --------------------------------------------------------------------------
# Experimental Caco-2 benchmark
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def caco2_experimental() -> pd.DataFrame:
    """Measured Caco-2 permeability of the benchmark natural products."""
    df = pd.read_csv(_require(C.FILE_CACO2_EXPERIMENTAL, _ZENODO_HINT))
    return df


@lru_cache(maxsize=1)
def caco2_predictions() -> pd.DataFrame:
    """Three-tool Caco-2 / P-gp predictions for the benchmark compounds.

    ADMET-AI writes a large negative value when its Caco-2 head fails; those are
    dropped here rather than treated as extremely low permeability.
    """
    df = pd.read_csv(_require(C.FILE_CACO2_PREDICTIONS, _ZENODO_HINT))
    df.loc[df["caco2__admet_ai"] <= C.AI_SENTINEL, "caco2__admet_ai"] = np.nan
    return df


def caco2_benchmark() -> pd.DataFrame:
    """Measurements joined to predictions, restricted to a usable Papp (A->B).

    Compounds are matched on the 14-character InChIKey connectivity block, which
    survives the re-canonicalization the web tools apply to submitted SMILES.
    """
    df = caco2_experimental().merge(caco2_predictions(), on="inchikey14", how="inner")
    df = df[df["papp_ab_cm_s"] > 0].copy()
    df["measured_log"] = np.log10(df["papp_ab_cm_s"])
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# NPASS 3.0 library
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def crosswalk() -> pd.DataFrame:
    """Endpoint-harmonization crosswalk: one row per (canonical endpoint, tool)."""
    return pd.read_csv(_require(C.FILE_CROSSWALK, _ZENODO_HINT))


@lru_cache(maxsize=1)
def canonical_endpoints() -> pd.DataFrame:
    """One row per canonical endpoint, derived from the crosswalk."""
    cw = crosswalk()
    out = (cw.groupby("endpoint", as_index=False)
             .agg(category=("category", "first"),
                  value_type=("value_type", "first"),
                  canonical_unit=("canonical_unit", "first"),
                  comparable=("comparable", "first"),
                  n_sources=("tool", "nunique")))
    out["comparable"] = out["comparable"].astype(bool)
    return out.rename(columns={"endpoint": "code"})


@lru_cache(maxsize=1)
def library_columns() -> tuple[str, ...]:
    """Column names of the library table, read without loading it."""
    head = pd.read_csv(_require(C.FILE_LIBRARY, _ZENODO_HINT), nrows=0)
    return tuple(head.columns)


def library(columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Harmonized predictions for the 201,905 standardized NPASS 3.0 compounds.

    ``columns`` restricts the read; the gzipped table is ~150 MB, so passing only
    what an analysis needs keeps memory and time reasonable.
    """
    path = _require(C.FILE_LIBRARY, _ZENODO_HINT)
    if columns is None:
        return _library_all()
    available = set(library_columns())
    use = [c for c in columns if c in available]
    return pd.read_csv(path, usecols=use)


@lru_cache(maxsize=1)
def _library_all() -> pd.DataFrame:
    return pd.read_csv(_require(C.FILE_LIBRARY, _ZENODO_HINT))


@lru_cache(maxsize=1)
def classification() -> pd.DataFrame:
    """NPClassifier assignments for the library, reduced to a primary pathway."""
    df = pd.read_csv(_require(C.FILE_CLASSIFICATION, _ZENODO_HINT),
                     usecols=["np_id", "pathway"], dtype=str)
    df["pathway"] = primary(df["pathway"])
    return df


def primary(s: pd.Series) -> pd.Series:
    """NPClassifier may return several pathways; the first one is used throughout."""
    return s.fillna("").str.split(";").str[0].str.strip().replace("", "Unclassified")


# --------------------------------------------------------------------------
# Wang et al. (2016) drug-like Caco-2 reference
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def wang_caco2() -> pd.DataFrame:
    """Drug-like Caco-2 benchmark with the authors' own train/test split.

    Sheet ``SI1`` of the supporting information; ``Dataset`` starts with "Tr" for
    training and "Te" for the held-out test compounds.
    """
    si = pd.read_excel(_require(C.FILE_WANG, _WANG_HINT), sheet_name="SI1")
    si.columns = [c.strip() for c in si.columns]
    si["measured_logPapp"] = pd.to_numeric(si["logPapp"], errors="coerce")
    si["smiles"] = si["smi"].astype(str)
    si["split"] = si["Dataset"].astype(str).str[:2]
    keep = si["measured_logPapp"].notna() & si["smiles"].notna()
    return si.loc[keep, ["NO", "name", "smiles", "Dataset", "split", "measured_logPapp"]].reset_index(drop=True)


# --------------------------------------------------------------------------
# Kim et al. (2026) replicate-level measurements
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def kim2026_replicates() -> pd.DataFrame:
    """Table S1 of Kim et al. (2026): one row per replicate measurement.

    The table holds three replicates of each of 83 compounds as separate rows,
    which is what makes the choice between a row-level and a compound-level split
    consequential (manuscript section 3.7). Structures are not in that table, so
    compounds are matched to the benchmark by normalized name; rows that do not
    match are dropped and counted.

    Reading it needs ``python-docx``, which is in requirements.txt.
    """
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise DataUnavailable(
            "python-docx is not installed; `pip install python-docx` to run the "
            "replicate-level re-analysis") from exc

    doc = Document(_require(C.FILE_KIM, _KIM_HINT))
    if not doc.tables:
        raise DataUnavailable(f"{C.FILE_KIM.name} contains no tables")

    rows = []
    for r in doc.tables[0].rows[1:]:
        cells = [c.text.strip() for c in r.cells]
        if len(cells) < 2 or not cells[0]:
            continue
        try:
            papp = float(cells[1])
        except ValueError:
            continue
        rows.append({"compound_name": cells[0], "papp": papp})

    df = pd.DataFrame(rows)
    df["key"] = df["compound_name"].map(normalize_name)

    exp = caco2_experimental().copy()
    exp["key"] = exp["compound_name"].map(normalize_name)
    df["smiles"] = df["key"].map(exp.drop_duplicates("key").set_index("key")["smiles"])
    return df.reset_index(drop=True)


def normalize_name(name: str) -> str:
    """Lower-case alphanumerics only, for matching compound names across sources."""
    import re
    import unicodedata
    return re.sub(r"[^a-z0-9]", "",
                  unicodedata.normalize("NFKD", str(name)).lower().replace("\u2212", "-"))


@lru_cache(maxsize=1)
def wang_admetai_predictions() -> pd.DataFrame:
    """ADMET-AI run on the Wang benchmark (produced by ``python -m npadmet.predict_wang``)."""
    hint = ("run `python -m npadmet.predict_wang` (needs the optional `admet-ai` package) "
            "to generate it")
    return pd.read_csv(_require(C.FILE_WANG_PREDICTIONS, hint))


# --------------------------------------------------------------------------
# Availability report
# --------------------------------------------------------------------------

def availability() -> pd.DataFrame:
    """Which inputs are present, for the README-style status message."""
    rows = [
        ("caco2_experimental.csv", "Zenodo", C.FILE_CACO2_EXPERIMENTAL),
        ("caco2_predictions.csv", "Zenodo", C.FILE_CACO2_PREDICTIONS),
        ("endpoint_crosswalk.csv", "Zenodo", C.FILE_CROSSWALK),
        ("npass3_admet_harmonized.csv.gz", "Zenodo", C.FILE_LIBRARY),
        ("npass3_biosynthetic_classification.csv", "Zenodo", C.FILE_CLASSIFICATION),
        (C.FILE_WANG.name, "Wang et al. 2016 SI", C.FILE_WANG),
        (C.FILE_KIM.name, "Kim et al. 2026 SI (optional)", C.FILE_KIM),
        (C.FILE_WANG_PREDICTIONS.name, "generated (optional)", C.FILE_WANG_PREDICTIONS),
        (C.FILE_PUBLICATION_STATUS.name, "tracked in this repository", C.FILE_PUBLICATION_STATUS),
    ]
    return pd.DataFrame([{"file": n, "source": s, "present": p.exists()} for n, s, p in rows])
