"""Paths, constants and naming used throughout the verification suite."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT = ROOT / "output"

TOOLS = ("admet_ai", "admetlab3", "admetsar3")
TOOL_LABELS = {"admet_ai": "ADMET-AI", "admetlab3": "ADMETlab 3.0", "admetsar3": "admetSAR 3.0"}
TOOL_PAIRS = (("admet_ai", "admetlab3"), ("admet_ai", "admetsar3"), ("admetlab3", "admetsar3"))
PAIR_LABELS = ("AI-L3", "AI-S3", "L3-S3")

# --- analysis constants (identical to the manuscript) ----------------------
PERM_THRESHOLD = -5.15   # log10 cm/s; "permeable" above this. Crosswalk default.
AI_SENTINEL = -10.0      # ADMET-AI writes a large negative value on model failure
EFFLUX_THRESHOLD = 2.0   # measured efflux ratio >= 2 -> active efflux
AD_CUTOFF = 0.30         # conventional applicability-domain nearest-neighbour Tanimoto
N_BOOTSTRAP = 2000
SEED = 42
DOSE_UG_ML = 100.0       # uniform mass concentration used in the Caco-2 assay
MIN_PATHWAY_N_ACCURACY = 8     # per-pathway minimum for the accuracy analysis
MIN_PATHWAY_N_AGREEMENT = 500  # per-pathway minimum for the library agreement analysis
MIN_N_AGREEMENT = 30           # minimum complete cases for a per-pathway agreement value

# --- revision analyses -----------------------------------------------------
MW_SPLIT = 500.0               # Da; the rule-of-five boundary used to stratify
AD_THRESHOLDS = (0.20, 0.25, 0.30, 0.35, 0.40)   # domain cutoffs for the sensitivity check
MIN_N_SUBGROUP = 8             # smallest subgroup that gets its own metrics
# Thresholds on the TEER change across the transport window, the standard
# monolayer-integrity check. -20% is the criterion most commonly used to discard
# a Caco-2 well; the benchmark itself is unfiltered (see section 3.10).
TEER_THRESHOLDS = (None, -50.0, -30.0, -20.0, -10.0)
TEER_COL = "teer_change_pct_0_2h"
N_FOLDS = 5                    # folds for every cross-validation reported
N_TREES = 500                  # random forest size for the reference model
MIN_LEAF = 2                   # random forest minimum leaf size

# --- data files -----------------------------------------------------------
# all downloaded from Zenodo (see data/README.md or `python download_data.py`)
FILE_CACO2_EXPERIMENTAL = DATA / "caco2_experimental.csv"
FILE_CACO2_PREDICTIONS = DATA / "caco2_predictions.csv"
FILE_LIBRARY = DATA / "npass3_admet_harmonized.csv.gz"
FILE_CLASSIFICATION = DATA / "npass3_biosynthetic_classification.csv"
FILE_CROSSWALK = DATA / "endpoint_crosswalk.csv"
# downloaded from the Wang et al. (2016) supporting information
FILE_WANG = DATA / "wang2016_caco2.xlsx"
# downloaded from the Kim et al. (2026) supporting information; only needed for the
# replicate-level re-analysis of section 3.7
FILE_KIM = DATA / "kim2026_tableS1.docx"
# optional, produced by `python -m npadmet.predict_wang`
FILE_WANG_PREDICTIONS = DATA / "wang_admetai_predictions.csv"
# tracked in this repository, not part of the Zenodo record: which benchmark
# compounds appear in the published tables of the source papers. Derived by name
# matching against those papers, not an experimental value, so it is versioned with
# the code rather than archived with the data.
FILE_PUBLICATION_STATUS = DATA / "benchmark_publication_status.csv"

# The open-data record. `download_data.py` resolves it through the Zenodo records
# API and fetches ZENODO_FILES from it.
ZENODO_CONCEPT_DOI = "10.5281/zenodo.22885507"
ZENODO_RECORD = "22885507"
ZENODO_DOI = ZENODO_CONCEPT_DOI
ZENODO_FILES = (FILE_CACO2_EXPERIMENTAL.name, FILE_CACO2_PREDICTIONS.name,
                FILE_CROSSWALK.name, FILE_LIBRARY.name, FILE_CLASSIFICATION.name)

WANG_URL = "https://pubs.acs.org/doi/10.1021/acs.jcim.5b00642"
KIM_URL = "https://doi.org/10.1002/jsfa.70701"
