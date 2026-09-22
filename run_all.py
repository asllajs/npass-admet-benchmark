"""Recompute every quantitative result of the paper and write it to ``output/``.

    python run_all.py            # everything the downloaded data allows
    python run_all.py --quick    # skip the two slow library-scale steps

Each step writes a CSV table and, where the paper shows one, a PNG figure named
after what it contains (``fig_caco2_validation.png``, not ``fig4.png``). All
numbers are collected into ``output/results.json``, which is what the pytest
suite compares against the published values in ``expected_results.json``.

Steps whose input has not been downloaded are reported as skipped rather than
failing the run, so a partial download still produces whatever it can.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from npadmet import (agreement, benchmark, caco2, chemspace, classes, config as C,
                     consensus, coverage, domain, generalization, stratify)
from npadmet.datasets import DataUnavailable, availability


def _write(table, name: str) -> str:
    C.OUTPUT.mkdir(parents=True, exist_ok=True)
    path = C.OUTPUT / f"{name}.csv"
    table.to_csv(path, index=False)
    return str(path)


# --------------------------------------------------------------------------
# Steps. Each returns the summary dictionary that ends up in results.json.
# --------------------------------------------------------------------------

def step_coverage() -> dict:
    _write(coverage.coverage_table(), "table_library_coverage")
    return {"library": coverage.library_coverage(), "crosswalk": coverage.crosswalk_summary()}


def step_chemical_space() -> dict:
    df, evr, order = chemspace.chemical_space()
    chemspace.figure(df, evr, order)
    return chemspace.summarize(df, evr, order)


def step_agreement() -> dict:
    table = agreement.agreement_table()
    _write(table, "table_inter_tool_agreement")
    agreement.figure(table)
    return agreement.summarize(table)


def step_caco2() -> dict:
    table = caco2.validation_table()
    efflux = caco2.efflux_table()
    _write(table, "table_caco2_validation")
    _write(efflux, "table_efflux_pgp")
    caco2.figure_validation()
    caco2.figure_efflux()
    return caco2.summarize(table, efflux)


def step_consensus() -> dict:
    table = consensus.consensus_table()
    inter = consensus.inter_tool_correlation(consensus.complete_case())
    _write(table, "table_consensus")
    _write(inter, "table_inter_tool_correlation")
    consensus.figure(table)
    return consensus.summarize(table, inter)


def step_class_accuracy() -> dict:
    accuracy = classes.accuracy_table()
    kruskal_result = classes.accuracy_kruskal()
    _write(accuracy, "table_class_accuracy")
    classes.figure_accuracy()
    return {"accuracy_rows": len(accuracy), "kruskal_absolute_error": kruskal_result,
            "median_abs_err_range": [round(float(accuracy["median_abs_err"].min()), 2),
                                     round(float(accuracy["median_abs_err"].max()), 2)]}


def step_class_agreement() -> dict:
    mat = classes.agreement_by_pathway()
    _write(mat, "table_class_agreement")
    classes.figure_agreement(mat)
    return {"caco2_agreement_by_pathway":
            {r["pathway"]: round(float(r["caco2"]), 2) for _, r in mat.iterrows()}}


def step_domain() -> dict:
    gap = domain.domain_gap()
    err = domain.error_vs_distance()
    confound = domain.molecular_weight_confound()
    _write(gap, "table_applicability_domain")
    _write(err, "table_error_vs_distance")
    _write(confound, "table_mw_confound")
    domain.figure(gap)
    domain.figure_confound()
    try:
        positive_control = domain.wang_positive_control()
        _write(positive_control, "table_wang_positive_control")
    except DataUnavailable as exc:
        print(f"    (positive control skipped: {exc})")
        positive_control = None
    return domain.summarize(gap, err, confound, positive_control)


def step_mw_stratified() -> dict:
    strat = stratify.stratified_table()
    quart = stratify.quartile_table()
    part = stratify.partial_table()
    _write(strat, "table_mw_stratified")
    _write(quart, "table_mw_quartiles")
    _write(part, "table_mw_partial")
    stratify.figure()
    return stratify.summarize(strat, quart, part)


def step_domain_split() -> dict:
    split = domain.domain_split_table()
    sens = domain.threshold_sensitivity()
    quint = domain.error_quintiles()
    paths = domain.domain_by_pathway()
    _write(split, "table_domain_split")
    _write(sens, "table_domain_threshold_sensitivity")
    _write(quint, "table_domain_quintiles")
    _write(paths, "table_domain_by_pathway")
    domain.figure_split(split, quint)
    summary = domain.summarize_split(split, sens, quint, paths)
    try:
        overlap = domain.chemspace_overlap()
        _write(overlap, "table_chemspace_overlap")
        summary["chemspace_overlap"] = {
            r["population"]: {"median_nn": round(float(r["median_nn_tanimoto"]), 3),
                              "frac_below_0.30": round(float(r["frac_below_0.30"]), 3)}
            for _, r in overlap.iterrows()}
    except DataUnavailable as exc:
        print(f"    (chemical-space overlap skipped: {exc})")
    return summary


def step_generalization() -> dict:
    table = generalization.generalization_table()
    counts = generalization.scaffold_counts()
    _write(table, "table_scaffold_generalization")
    _write(generalization.learning_curve(), "table_learning_curve")
    try:
        protocols = generalization.replicate_protocols()
        spread = generalization.replicate_spread()
        _write(protocols, "table_replicate_protocols")
        _write(spread, "table_replicate_spread")
        _write(generalization.replicate_zero_fraction(), "table_replicate_zeros")
    except DataUnavailable as exc:
        print(f"    (replicate re-analysis skipped: {exc})")
        protocols = spread = None
    generalization.figure(table, protocols)
    return generalization.summarize(table, counts, protocols, spread)


def step_benchmark_context() -> dict:
    taut = benchmark.tautomer_table()
    effect = benchmark.tautomer_effect(taut)
    _write(taut, "table_tautomers")
    _write(effect, "table_tautomer_effect")
    try:
        het = benchmark.subset_heterogeneity()
        _write(het, "table_subset_heterogeneity")
    except DataUnavailable as exc:
        print(f"    (subset heterogeneity skipped: {exc})")
        het = None
    try:
        signal = benchmark.descriptor_signal()
        _write(signal, "table_descriptor_signal")
    except DataUnavailable as exc:
        print(f"    (descriptor signal skipped: {exc})")
        signal = None
    try:
        rep = benchmark.representativeness()
        _write(rep, "table_representativeness")
    except DataUnavailable as exc:
        print(f"    (representativeness skipped: {exc})")
        rep = None
    summary = benchmark.summarize(taut, effect, rep, het, signal)
    try:
        sens = benchmark.teer_sensitivity()
        comp = benchmark.teer_composition()
        tsig = benchmark.teer_descriptor_signal()
        _write(sens, "table_teer_sensitivity")
        _write(comp, "table_teer_composition")
        _write(tsig, "table_teer_descriptor_signal")
        summary["monolayer_integrity"] = benchmark.summarize_teer(sens, comp, tsig)
    except DataUnavailable as exc:
        print(f"    (monolayer-integrity sensitivity skipped: {exc})")
    return summary


STEPS = [
    ("library_coverage", step_coverage, False),
    ("chemical_space", step_chemical_space, True),
    ("inter_tool_agreement", step_agreement, True),
    ("caco2_validation", step_caco2, False),
    ("consensus", step_consensus, False),
    ("class_accuracy", step_class_accuracy, False),
    ("class_agreement", step_class_agreement, True),
    ("applicability_domain", step_domain, False),
    ("mw_stratified", step_mw_stratified, False),
    ("domain_split", step_domain_split, True),
    ("generalization", step_generalization, False),
    ("benchmark_context", step_benchmark_context, True),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="skip the slow steps that read the whole library table")
    args = parser.parse_args(argv)

    status = availability()
    print("Input files")
    for row in status.itertuples():
        print(f"  [{'x' if row.present else ' '}] {row.file}  ({row.source})")
    print()

    results, skipped = {}, []
    for name, fn, slow in STEPS:
        if args.quick and slow:
            skipped.append((name, "skipped by --quick"))
            print(f"- {name}: skipped (--quick)")
            continue
        started = time.time()
        try:
            results[name] = fn()
        except DataUnavailable as exc:
            skipped.append((name, str(exc)))
            print(f"- {name}: SKIPPED ({exc})")
            continue
        print(f"- {name}: done in {time.time() - started:.1f}s")

    C.OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {"results": results,
               "skipped": {name: reason for name, reason in skipped}}
    (C.OUTPUT / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True),
                                           encoding="utf-8")
    print(f"\nwrote {C.OUTPUT / 'results.json'} and {len(list(C.OUTPUT.glob('*')))} files "
          f"into {C.OUTPUT}")
    if skipped:
        print("\nNot run:")
        for name, reason in skipped:
            print(f"  {name}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
