"""Optional step: run ADMET-AI on the drug-like Wang (2016) Caco-2 benchmark.

This produces the in-domain positive control - the same tool, the same endpoint,
inside the chemistry it was trained on. It needs the optional ``admet-ai``
package, which pulls in a deep-learning stack, so it is kept out of the default
requirements and out of the default test run.

    pip install admet-ai
    python -m npadmet.predict_wang

Writes ``data/wang_admetai_predictions.csv``; afterwards
``tests/test_08_applicability_domain.py`` checks the positive control instead of
skipping it.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from . import datasets as D


def main() -> None:
    wang = D.wang_caco2()
    try:
        from admet_ai import ADMETModel
    except ImportError as exc:  # pragma: no cover - depends on an optional install
        raise SystemExit("this step needs the optional `admet-ai` package: pip install admet-ai") from exc

    model = ADMETModel()
    pred = model.predict(smiles=wang["smiles"].tolist())
    caco2_column = [c for c in pred.columns if "caco" in c.lower()][0]
    out = wang.copy()
    out["ai_caco2"] = np.asarray(pred[caco2_column])
    out.to_csv(C.FILE_WANG_PREDICTIONS, index=False)
    print(f"wrote {C.FILE_WANG_PREDICTIONS} ({len(out)} compounds)")


if __name__ == "__main__":
    main()
