# Data

Only this README and `benchmark_publication_status.csv` are tracked in git. Everything
this study produced comes from one Zenodo record. One further file, a third-party
reference set, has to be fetched by hand.

```
data/
├── benchmark_publication_status.csv          tracked in this repository
├── caco2_experimental.csv                    Zenodo
├── caco2_predictions.csv                     Zenodo
├── endpoint_crosswalk.csv                    Zenodo
├── npass3_admet_harmonized.csv.gz            Zenodo
├── npass3_biosynthetic_classification.csv    Zenodo
├── wang2016_caco2.xlsx                       Wang et al. (2016) supporting information
├── kim2026_tableS1.docx                      Kim et al. (2026) supporting information, optional
└── wang_admetai_predictions.csv              generated, optional
```

Run `python run_all.py` at any point: it prints which of these are present and runs
whatever the available data allows.

---

## 1. From Zenodo

```bash
python download_data.py
```

fetches the five files below (about 170 MB) from
**<https://doi.org/10.5281/zenodo.22885507>** (CC BY 4.0). To do it by hand, open the
record and save them into this folder.

The record also holds `npass3_admet_raw.csv.gz` (the untransformed native tool
outputs) and `npass3_admet.sqlite` (626 MB, the same content as a single queryable
database). Neither is needed here, so `download_data.py` leaves them alone.

### `caco2_experimental.csv` — 142 natural products

Measured Caco-2 permeability of the benchmark compounds: apical-to-basolateral and
basolateral-to-apical apparent permeability and the derived efflux ratio, measured on
a differentiated Caco-2 monolayer under a single protocol at a uniform 100 µg mL⁻¹.
134 compounds have a usable P<sub>app</sub> (A→B), 139 a P<sub>app</sub> (B→A) and
131 an efflux ratio. Natural-product status is verifiable against NPASS 3.0 once
counter-ions are stripped, which `tests/test_11_benchmark_provenance.py` checks.

| column | meaning |
|---|---|
| `inchikey14` | first 14 characters of the InChIKey; the join key used throughout |
| `inchikey`, `smiles`, `compound_name`, `pubchem_cid`, `mw` | structure and identity |
| `assay_chemical_class` | the chemical class used in the source permeability study |
| `npc_pathway`, `npc_superclass`, `npc_class` | NPClassifier biosynthetic assignment |
| `papp_ab_cm_s`, `papp_ba_cm_s` | apparent permeability, cm s⁻¹ |
| `efflux_ratio` | P<sub>app</sub>(B→A) / P<sub>app</sub>(A→B) |
| `teer_change_pct_0_2h` | percentage change in transepithelial electrical resistance over the first 2 h of exposure; used in Section 3.10 to flag compromised monolayers |

The parts of these measurements published so far appear in Cong et al. (2026),
*Food & Function* 17:2955–2970, <https://doi.org/10.1039/D5FO05099E>, and
Kim et al. (2026), *J. Sci. Food Agric.* 106:6606–6616,
<https://doi.org/10.1002/jsfa.70701>.

### `caco2_predictions.csv` — 142 natural products

The three tools' outputs for the same compounds, on the harmonized scale. Both
endpoints kept here carry the transform `none` in the crosswalk, so the harmonized
values are the tools' native outputs.

| column | meaning |
|---|---|
| `caco2__admet_ai`, `caco2__admetlab3`, `caco2__admetsar3` | predicted log P<sub>app</sub>, log cm s⁻¹ |
| `pgp_substrate__admetlab3`, `pgp_substrate__admetsar3` | P-glycoprotein substrate probability |
| `pgp_inhibitor__admet_ai` | P-gp *inhibition* probability (ADMET-AI has no substrate head) |

ADMET-AI writes a large negative sentinel when its Caco-2 head fails; the loader
drops those rather than reading them as very low permeability.

### The library tables

| file | size | contents |
|---|---|---|
| `npass3_admet_harmonized.csv.gz` | 149 MB | one row per compound: identity, NPClassifier class, per-tool coverage flags, and the harmonized value of every endpoint × tool (`<endpoint>__<tool>`) |
| `npass3_biosynthetic_classification.csv` | 19 MB | NPClassifier assignments for the whole library |
| `endpoint_crosswalk.csv` | 9 kB | the harmonization crosswalk, manuscript Table 2 |

---

## 2. From the publisher — the drug-like reference set

`wang2016_caco2.xlsx` is the supporting-information spreadsheet of

> Wang N-N, Dong J, Deng Y-H, Zhu M-F, Wen M, Yao Z-J, Lu A-P, Wang J-B, Cao D-S
> (2016) ADME properties evaluation in drug discovery: prediction of Caco-2 cell
> permeability using a combination of NSGA-II and boosting. *J Chem Inf Model*
> 56:763–773. <https://doi.org/10.1021/acs.jcim.5b00642>

It is the source of the Therapeutics Data Commons `Caco2_Wang` task that ADMET-AI's
Caco-2 head is trained on, and it is used here for two things: the nearest-neighbour
distance that defines the applicability domain, and the in-domain positive control.
The analyses read sheet **`SI1`**, which carries the columns `smi`, `logPapp` and
`Dataset` (values beginning `Tr` = training, `Te` = held-out test).

It is third-party supporting information and is therefore not redistributed: open the
article page, download the `.xlsx` from Supporting Information, and save it here as
`wang2016_caco2.xlsx`.

---

## 3. From Kim et al. (2026) — optional

Section 3.7 of the paper re-evaluates the replicate-level measurements behind the
natural-product permeability model it discusses. Those measurements are Table S1 of

> Kim J-W, Cong R, Park J-S, Park K, Kang K, Shim S-M (2026) Molecular descriptor
> driven QSPR modeling of Papp, TEER and efflux ratio from Caco-2 cells using
> machine learning for various phytochemicals. *J Sci Food Agric* 106:6606–6616.
> <https://doi.org/10.1002/jsfa.70701>

Download the supporting-information document from that article and save it here as
`kim2026_tableS1.docx`. It is third-party material and is not redistributed with
this repository.

Reading it needs `python-docx`, which is in `requirements.txt`. Without the file,
the replicate-level tests in `test_14_generalization.py` skip with an explicit
reason and every other analysis still runs.

The table holds 248 rows for 83 compounds — three replicates each, one row per
replicate — and 91 of those rows record a permeability of exactly zero. Both facts
matter for how a cross-validated accuracy on this panel should be read, which is
the point the section makes.

---

## 4. Generated — the positive control (optional)

`wang_admetai_predictions.csv` holds ADMET-AI's Caco-2 predictions for the Wang
compounds. Producing it means running the model:

```bash
pip install admet-ai
python -m npadmet.predict_wang
```

Without it, `tests/test_08_applicability_domain.py` skips the two positive-control
tests and reports why; the applicability-domain distance itself does not need it.

---

## 5. Tracked here — which benchmark compounds are already in print

`benchmark_publication_status.csv` is the one data file that ships with the code.
Section 3.3 and Table S9 split the benchmark by whether a compound's measured
permeability has already been tabulated in a paper from the same assay programme:

> Cong et al. (2026) *Food Funct.* 17:2955–2970, Table 2 ·
> Kim et al. (2026) *J Sci Food Agric* 106:6606–6616, Table S1 ·
> Lee et al. (2025) *J Sci Food Agric* 105:6243–6253, Table 2

| column | meaning |
|---|---|
| `inchikey14` | join key, as in `caco2_experimental.csv` |
| `compound_name` | name used for the matching |
| `in_cong`, `in_kim`, `in_lee` | whether the compound appears in that paper's table |
| `published` | any of the three |

The flag is derived by matching compound names against those papers, so it is an
annotation rather than a measurement; that is why it is versioned with the code
instead of archived on Zenodo. Name matching is imperfect for salt forms and
alternative names.

The 140 rows are the assay-panel compounds with a measured P<sub>app</sub> (A→B):
the 134 benchmark compounds that enter Table S9 (100 tabulated, 34 not), plus six
panel compounds outside the natural-product benchmark, which the loader ignores.
The eight benchmark compounds with only a B→A measurement have no row and do not
enter Table S9 in any case. `npadmet/benchmark.py` reads the file and
`tests/test_15_benchmark_context.py` checks the result.

---

## Licence

The Zenodo files are released under **CC BY 4.0**. The code in this repository is
MIT-licensed; see `../LICENSE`.
