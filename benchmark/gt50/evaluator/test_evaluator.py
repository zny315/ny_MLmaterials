from __future__ import annotations

import importlib.util
from collections import defaultdict
from pathlib import Path

import pandas as pd


WRAPPER = Path(__file__).with_name("evaluate_gt50.py")


def load_wrapper():
    spec = importlib.util.spec_from_file_location("final_evaluator_v15", WRAPPER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_material_subtypes(wrapper, scorer) -> None:
    assert wrapper.strong_material_equal(scorer, "PE", "polyethylene")[0]
    assert wrapper.strong_material_equal(scorer, "LDPE", "low-density polyethylene")[0]
    assert not wrapper.strong_material_equal(scorer, "PE", "LDPE")[0]
    assert not wrapper.strong_material_equal(scorer, "LDPE", "LLDPE")[0]
    assert not wrapper.strong_material_equal(scorer, "Cloisite 15A", "Cloisite 30B")[0]
    assert wrapper.scored_material_equal(scorer, "PE", "VLDPE")[0]
    assert not wrapper.scored_material_equal(scorer, "VLDPE", "PE")[0]
    assert not wrapper.scored_material_equal(
        scorer,
        "alumina-supported Co/Fe catalyst residue",
        "alumina",
    )[0]
    assert not wrapper.scored_material_equal(scorer, "sodium stearate", "stearate")[0]
    assert not wrapper.scored_material_equal(
        scorer,
        "oligomeric vinyl silane",
        "oligomeric vinyl silane with boric acid",
    )[0]
    assert wrapper.scored_material_equal(scorer, "SMR L", "SMR L natural rubber")[0]
    gt_row = {scorer.COL["entity_name"]: "LDPE", wrapper.PE_TYPE_FIELD: "LDPE"}
    pred_row = {scorer.COL["entity_name"]: "PE", wrapper.PE_TYPE_FIELD: "LDPE"}
    assert wrapper.entity_name_rows_equal(scorer, gt_row, pred_row)[0]
    pred_row[wrapper.PE_TYPE_FIELD] = "PE"
    assert not wrapper.entity_name_rows_equal(scorer, gt_row, pred_row)[0]


def loading_row(wrapper, normalized: str, wt_origin: str, phr_origin: str) -> dict:
    return {
        wrapper.NORMALIZED_LOADING_FIELD: normalized,
        wrapper.LOADING_WT_ORIGIN: wt_origin,
        wrapper.LOADING_PHR_ORIGIN: phr_origin,
    }


def test_loading_provenance(wrapper, scorer) -> None:
    both_derived_left = loading_row(
        wrapper,
        "wt=70;phr=100",
        "explicit",
        "derived_from_wt_and_matrix_wt",
    )
    both_derived_right = loading_row(
        wrapper,
        "wt=55;phr=100",
        "explicit",
        "derived_from_wt_and_matrix_wt",
    )
    assert not wrapper.loading_rows_equal(scorer, both_derived_left, both_derived_right)[0]

    cross_basis_left = loading_row(
        wrapper,
        "wt=70;phr=100",
        "explicit",
        "derived_from_wt_and_matrix_wt",
    )
    cross_basis_right = loading_row(
        wrapper,
        "wt=70;phr=100",
        "derived_from_phr_and_matrix_wt",
        "explicit",
    )
    assert wrapper.loading_rows_equal(scorer, cross_basis_left, cross_basis_right)[0]

    explicit_phr_left = loading_row(wrapper, "wt=40;phr=20", "derived", "explicit")
    explicit_phr_right = loading_row(wrapper, "wt=42;phr=20", "derived", "explicit")
    assert wrapper.loading_rows_equal(scorer, explicit_phr_left, explicit_phr_right)[0]


def test_verified_scales(wrapper, scorer) -> None:
    assert wrapper.strict_numeric_equal(scorer, "0.056", "5.6", "pSPR", "50")[0]
    assert not wrapper.strict_numeric_equal(scorer, "0.056", "5.6", "pSPR", "51")[0]
    assert wrapper.strict_numeric_equal(scorer, "13", "0.013", "pCOR", "524")[0]
    assert not wrapper.strict_numeric_equal(scorer, "13", "0.013", "pCOR", "523")[0]


def performance_row(scorer, object_uid: str, sample_uid: str, sample_name: str, loi: str, phrr: str):
    row = {field: "" for field in scorer.PERFORMANCE_FIELDS}
    row.update(
        {
            scorer.COL["article"]: "1",
            scorer.COL["sample_name"]: sample_name,
            "_performance_uid": object_uid,
            "_linked_sample_uid": sample_uid,
            "LOI": loi,
            "pHRR": phrr,
        }
    )
    return row


def test_cross_sample_swap(wrapper, scorer) -> None:
    scorer.numeric_equal = lambda left, right, field: wrapper.strict_numeric_equal(
        scorer, left, right, field
    )
    scorer.field_equal = wrapper.strict_field_equal(scorer, scorer.field_equal)
    scorer.score_pair = wrapper.strict_score_pair(scorer)
    scorer.score_performance = wrapper.score_performance(scorer)
    gt = pd.DataFrame(
        [
            performance_row(scorer, "gt-A", "gt-A", "Sample A", "20", "100"),
            performance_row(scorer, "gt-B", "gt-B", "Sample B", "30", "200"),
        ]
    )
    prediction = pd.DataFrame(
        [
            performance_row(scorer, "pred-A", "pred-A", "Sample A", "30", "200"),
            performance_row(scorer, "pred-B", "pred-B", "Sample B", "20", "100"),
        ][::-1]
    )
    methods = {"Synthetic": {"performance": prediction}}
    sample_mappings = {"Synthetic": {"1": {"gt-A": "pred-A", "gt-B": "pred-B"}}}
    sample_aliases = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    records = scorer.score_performance(gt, methods, sample_mappings, sample_aliases)
    scored = [record for record in records if record["field"] in {"LOI", "pHRR"}]
    assert len(scored) == 4
    assert all(record["outcome"] == "FP+FN" for record in scored)


def main() -> None:
    wrapper = load_wrapper()
    scorer = wrapper.load_base_module()
    test_material_subtypes(wrapper, scorer)
    test_loading_provenance(wrapper, scorer)
    test_verified_scales(wrapper, scorer)
    test_cross_sample_swap(wrapper, scorer)
    print("final evaluator v18 regression tests passed")


if __name__ == "__main__":
    main()
