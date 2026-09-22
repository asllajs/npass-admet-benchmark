# npass-admet-benchmark

Verification suite for the paper

> **Benchmarking the mutual agreement and experimental Caco-2 accuracy of three
> modern ADMET predictors across natural products**
> Yang J-S, Park K, Kim E, Park J-S, Hwang I, Shim S-M, Kang K. (manuscript under review)

The study benchmarks **ADMET-AI**, **ADMETlab 3.0** and **admetSAR 3.0** across the
201,905 standardized natural products of NPASS 3.0, harmonizes 33 endpoints onto a
common scale, and validates the Caco-2 permeability predictions against measured
apparent permeability for 134 structurally diverse natural products. Its
conclusion is that the three tools agree closely with one another and not at all
with experiment, that this is neither a dosing artefact nor a scaffold-novelty
effect, and that a conventional applicability-domain flag does not identify the
predictions that go wrong.

This repository is not a figure-rendering pipeline. It is a **test suite for the
paper's results**: every quantitative claim is recomputed from the open data and
asserted against the published value, so a reader can run one command and see
whether the conclusions still hold.

```
$ pytest
================== 197 passed in 234s ==================
```

Each test is named after the claim it checks, and the failure message tells you
which published number moved. Alongside the assertions, `run_all.py` writes the
underlying tables and a plain figure for each one into `output/`.

---

## Quick start

```bash
git clone https://github.com/asllajs/npass-admet-benchmark.git
cd npass-admet-benchmark
pip install -r requirements.txt

python download_data.py     # ~170 MB from Zenodo (10.5281/zenodo.22885507)
pytest                      # verify the published numbers
python run_all.py           # write the tables and figures into output/
```

`python run_all.py` starts by listing which input files it can see, so it is also
the quickest way to check a partial download. One extra file has to be fetched by
hand for the applicability-domain analysis — see [`data/README.md`](data/README.md).

To skip the steps that read the 150 MB library table:

```bash
pytest -m "not slow"        # 147 tests
```

The revision analyses (`test_12` to `test_15`) need one extra third-party file for
the replicate-level re-analysis of section 3.7 — see
[`data/README.md`](data/README.md). Without it those tests skip with an explicit
reason; everything else still runs.

---

## What is checked

| Test file | Claim under test | Needs |
|---|---|---|
| `test_01_library_coverage.py` | 201,905 compounds; per-tool coverage 100.0 / 94.6 / 98.2 %; 92.8 % covered by all three | Zenodo |
| `test_02_crosswalk.py` | 91 native endpoints map onto 33 canonical ones; 30 comparable across ≥ 2 tools; half-life, clearance and acute toxicity are flagged as not comparable | Zenodo |
| `test_03_inter_tool_agreement.py` | deterministic descriptors agree at CCC ≈ 1.00 while ADMET endpoints do not; Caco-2 three-way CCC 0.59; CYP2C9-substrate κ ≈ 0 | Zenodo |
| `test_04_caco2_validation.py` | no tool tracks the measured permeability (r = 0.05–0.10, all p > 0.05, R² < 0, AUROC 0.54–0.61) | Zenodo |
| `test_05_consensus.py` | the tools correlate with each other at r ≈ 0.88 and with experiment at r ≈ 0.09, and consensus does not beat any single tool | Zenodo |
| `test_06_efflux.py` | predicted P-gp substrate probability is uninformative of the measured efflux ratio | Zenodo |
| `test_07_class_stratified.py` | agreement is class-dependent (0.83 for shikimates down to 0.22 for peptides), error is not (Kruskal–Wallis p = 0.10 to 0.54) | Zenodo |
| `test_08_applicability_domain.py` | the benchmark sits outside the drug-like training domain (median nearest-neighbour Tanimoto 0.39 vs 0.77) and distance does not triage individual compounds | Zenodo + Wang SI |
| `test_09_chemical_space.py` | the projection covers all 195,098 classified compounds; PC1 + PC2 = 90 % of variance | Zenodo |
| `test_10_headline_claims.py` | the seven statements of the abstract, one test each | all |
| `test_11_benchmark_provenance.py` | every benchmark compound is catalogued in NPASS 3.0 once counter-ions are stripped | Zenodo |
| `test_12_mw_stratified.py` | the null result survives a split at 500 Da, every molecular-weight quartile, and a partial correlation holding molecular weight fixed | Zenodo |
| `test_13_domain_split.py` | 98 of 134 compounds are in-domain and predicted no better; no cutoff from 0.20 to 0.40 changes that; error does not rank with distance | Zenodo + Wang SI |
| `test_14_generalization.py` | a fixed model keeps R² 0.57 and AUROC 0.83 under a Bemis–Murcko scaffold split and falls to −1.85 and chance on transfer; the replicate-level panel loses its reported accuracy under a compound-level split, on both value scales | Zenodo + Wang SI (+ Kim SI) |
| `test_15_benchmark_context.py` | tautomerism does not explain the error; the benchmark is closer to drug-like chemistry than the library; its two halves disagree | Zenodo |

Reference values live in [`expected_results.json`](expected_results.json), one
block per manuscript section, with the tolerance for each comparison written next
to the assertion that uses it.

A test whose input has not been downloaded **skips with an explicit reason**
rather than passing quietly:

```
SKIPPED [3] missing npass3_admet_harmonized.csv.gz in data - run `python download_data.py`
```

---

## What is produced

`python run_all.py` writes everything into `output/`. Files are named after their
contents, not after a figure number, and all figures are PNG.

| File | Manuscript equivalent |
|---|---|
| `fig_chemical_space.png` | Fig. 2 — chemical space and Fsp3 by pathway |
| `fig_caco2_validation.png` | Fig. 3 — measured vs predicted, and Bland–Altman |
| `fig_consensus.png` | Fig. 4 — consensus against experiment |
| `fig_mw_stratified.png` | Fig. 5 — measured vs predicted, stratified at 500 Da |
| `fig_domain_split.png` | Fig. 6 — distance from the training domain |
| `fig_scaffold_generalization.png` | Fig. 7 — scaffold-disjoint and cross-domain transfer |
| `fig_model_agreement.png` | Fig. 8 — inter-tool agreement per endpoint |
| `fig_class_accuracy.png` | Fig. 9a — error by biosynthetic pathway |
| `fig_class_agreement.png` | Fig. 9b — agreement by pathway × endpoint |
| `fig_efflux_pgp.png` | Fig. S1 — P-gp prediction vs measured efflux |
| `fig_mw_confound.png` | Fig. S2a–c — error against molecular weight |
| `fig_applicability_domain.png` | supporting view of the domain analysis |
| `table_library_coverage.csv` | Section 3.1 |
| `table_caco2_validation.csv` | Table 3 (upper block) |
| `table_teer_sensitivity.csv` | Table 3 (lower block) and Table S4 |
| `table_consensus.csv` | Table 4 |
| `table_mw_stratified.csv` | Table 5 |
| `table_domain_split.csv` | Table 6 |
| `table_scaffold_generalization.csv` | Table 7 |
| `table_representativeness.csv` | Table S1 |
| `table_chemspace_overlap.csv` | Table S2 |
| `table_tautomers.csv` | Table S3 |
| `table_domain_threshold_sensitivity.csv` | Table S5 |
| `table_domain_by_pathway.csv` | Table S6 |
| `table_class_agreement.csv` | Table S7 |
| `table_class_accuracy.csv` | Table S8 |
| `table_subset_heterogeneity.csv` | Table S9 |
| `table_efflux_pgp.csv`, `table_error_vs_distance.csv`, `table_domain_quintiles.csv`, `table_mw_confound.csv`, `table_mw_partial.csv`, `table_mw_quartiles.csv`, `table_learning_curve.csv`, `table_replicate_*.csv`, `table_tautomer_effect.csv`, `table_teer_composition.csv`, `table_teer_descriptor_signal.csv`, `table_descriptor_signal.csv`, `table_inter_tool_agreement.csv`, `table_inter_tool_correlation.csv`, `table_wang_positive_control.csv` | supporting analyses cited in the text |
| `results.json` | every number above, in one file |

The figures use stock matplotlib with its default font and no style sheet, so
they render identically on a clean install. They are meant to be legible, not to
be the typeset figures of the article; the numbers, not the styling, are what
this repository is for.

---

## Layout

```
npadmet/
  config.py       paths, tool names, thresholds (one place for every constant)
  datasets.py     loading, in three tiers, with explicit errors when a file is absent
  metrics.py      CCC, Cohen's and Fleiss' kappa, AUROC, MCC, bootstrap CI, identity-line R²
  coverage.py     section 3.1
  chemspace.py    Fig. 2
  agreement.py    section 3.8
  caco2.py        section 3.2, Table 3
  consensus.py    section 3.3, Table 4
  classes.py      section 3.9
  domain.py       section 3.5 and the two controls; Table 6 and Tables S2, S5, S6
  stratify.py     section 3.4, Table 5: the molecular-weight stratification
  generalization.py  sections 3.6 and 3.7, Table 7: scaffold split, cross-domain
                  transfer and the replicate-level re-analysis
  benchmark.py    sections 2.1, 2.4 and 3.10: what the benchmark is made of;
                  Tables S1, S3, S9 and the monolayer-integrity sensitivity
  plots.py        plain matplotlib helpers
  predict_wang.py optional: runs ADMET-AI for the positive control
tests/            one file per manuscript section
data/             inputs, downloaded (see data/README.md); one file is tracked here,
                  benchmark_publication_status.csv, the annotation behind Table S9
run_all.py        recompute everything into output/
download_data.py  fetch the Zenodo half of the inputs
```

The analysis functions return DataFrames and never plot as a side effect, so they
can be imported and reused:

```python
from npadmet import caco2, consensus

caco2.validation_table()          # manuscript Table 3
consensus.consensus_table()       # manuscript Table 4
```

Statistical choices follow the manuscript exactly: agreement between predictions
by Lin's concordance correlation coefficient for regression endpoints and Cohen's
or Fleiss' κ for classification endpoints; agreement with experiment by Pearson
and Spearman correlation, RMSE, identity-line R², CCC and Bland–Altman limits;
binary permeability at −5.15 log cm s⁻¹ applied identically to measured and
predicted values; 95 % confidence intervals from 2,000 bootstrap replicates with
a fixed seed.

---

## Data

Summarized here, in full in [`data/README.md`](data/README.md).

- **Zenodo, <https://doi.org/10.5281/zenodo.22885507>** (CC BY 4.0) — everything
  this study produced: the experimental Caco-2 measurements for 142 natural products
  and the three tools' predictions for them, the harmonized predictions for the whole
  NPASS 3.0 library, the biosynthetic classification and the endpoint crosswalk.
  `python download_data.py`.
- **Wang et al. (2016) supporting information** — the drug-like Caco-2 reference
  set, downloaded by hand from the publisher.
- **Tracked in this repository** — `data/benchmark_publication_status.csv`, which
  benchmark compounds are tabulated in a source paper (the split behind Table S9).
  An annotation derived by name matching, not a measurement, so it is versioned
  with the code rather than archived on Zenodo.

---

## Citing

Please cite the article and the Zenodo data record
(<https://doi.org/10.5281/zenodo.22885507>).

The benchmarked tools are third-party software: ADMET-AI (Swanson et al., 2024),
ADMETlab 3.0 (Fu et al., 2024), admetSAR 3.0 (Gu et al., 2024). Structures come
from NPASS 3.0 (Lin et al., 2026) and biosynthetic classes from NPClassifier
(Kim et al., 2021). The web-hosted predictors were queried in May 2026; ADMET-AI
was run as a local Python package.

## Licence

Code: MIT. Data: CC BY 4.0. See `LICENSE`.
