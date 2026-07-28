from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import pandas as pd


SOURCE_SCORER = Path(__file__).with_name("hierarchical_score_base.py")
GT_PATH: Path
OUTPUT_DIR: Path
DIRECT_DIR: Path
EXPERT_DIR: Path
SPLIT_DIR: Path
FULL_DIR: Path
RERUN_ROOT: Path | None = None
OLD_ALIGNMENT: Path | None = None
OLD_NEGATIVE_57: Path | None = None
SECONDARY_CONDITION_PATTERN = (
    r"water[ -]?treat|thermal[ -]?ag|aged\s+(?:for|at)"
)


RELATION_ALIASES = {
    "treated_surface_of": "treated_surface_of",
    "surface_treated_by": "treated_surface_of",
    "grafted_to": "grafted_to",
    "grafted_from": "grafted_to",
    "intercalated_into": "intercalated_into",
    "intercalated_in": "intercalated_into",
    "doped_into": "doped_into",
    "doped_in": "doped_into",
}

METHOD_ALIASES = {
    "microencapsulation": "encapsulation",
    "coordination_complex": "complexation",
    "physical_adsorption": "surface_treatment",
}

MODEL_TOKEN_RE = re.compile(r"(?i)\b[a-z]*\d+[a-z0-9.-]*\b")
POLYETHYLENE_SUBTYPE_PATTERNS = [
    ("lldpe", re.compile(r"(?i)\blldpe\b|linear\s+low[-\s]+density\s+polyethylene")),
    ("vldpe", re.compile(r"(?i)\bvldpe\b|very\s+low[-\s]+density\s+polyethylene")),
    ("hdpe", re.compile(r"(?i)\bhdpe\b|high[-\s]+density\s+polyethylene")),
    ("ldpe", re.compile(r"(?i)\bldpe\b|low[-\s]+density\s+polyethylene")),
    ("pe", re.compile(r"(?i)\bpe\b|\bpolyethylene\b")),
]
IDENTITY_QUALIFIER_PATTERNS = [
    ("with", re.compile(r"(?i)\bwith\b")),
    ("supported", re.compile(r"(?i)\bsupported\b")),
    ("catalyst", re.compile(r"(?i)\bcatalyst\b")),
    ("residue", re.compile(r"(?i)\bresidue\b")),
    ("sodium", re.compile(r"(?i)\bsodium\b")),
    ("potassium", re.compile(r"(?i)\bpotassium\b")),
    ("calcium", re.compile(r"(?i)\bcalcium\b")),
    ("zinc", re.compile(r"(?i)\bzinc\b")),
    ("magnesium", re.compile(r"(?i)\bmagnesium\b")),
    ("aluminum", re.compile(r"(?i)\balumin(?:um|ium)\b")),
    ("cobalt", re.compile(r"(?i)\bcobalt\b|\bco\s*/")),
    ("iron", re.compile(r"(?i)\biron\b|/\s*fe\b")),
    ("copper", re.compile(r"(?i)\bcopper\b")),
    ("nickel", re.compile(r"(?i)\bnickel\b")),
    ("boric_acid", re.compile(r"(?i)\bboric\s+acid\b")),
]
LOADING_WT_ORIGIN = "_loading_wt_origin"
LOADING_PHR_ORIGIN = "_loading_phr_origin"
NORMALIZED_LOADING_FIELD = "entity_loading_normalized_wt_or_phr"
PE_TYPE_FIELD = "PE_type"
UL94_FIELD = "UL94_rating"
VERIFIED_SCALE_FACTORS = {
    ("38", "cone_specimen_thickness"): (Decimal("10"),),
    ("50", "pSPR"): (Decimal("100"),),
    ("234", "pCOR"): (Decimal("1000"),),
    ("234", "pCO2R"): (Decimal("1000"),),
    ("524", "pCOR"): (Decimal("1000"),),
}
CONDITION_PATTERNS = {
    "loading_wt": re.compile(r"(?i)(\d+(?:\.\d+)?)\s*wt\.?\s*%"),
    "loading_pct": re.compile(r"(?i)(\d+(?:\.\d+)?)\s*%(?!\s*(?:va|vinyl acetate))"),
    "loading_phr": re.compile(r"(?i)(\d+(?:\.\d+)?)\s*phr\b"),
    "irradiation_kgy": re.compile(r"(?i)(\d+(?:\.\d+)?)\s*kgy\b"),
    "time_s": re.compile(r"(?i)(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b"),
    "time_h": re.compile(r"(?i)(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\b"),
    "temperature_c": re.compile('(?i)(\\d+(?:\\.\\d+)?)\\s*(?:\xb0\\s*c|\u2103|deg(?:ree)?s?\\s*c)\\b'),
}


def load_base_module():
    spec = importlib.util.spec_from_file_location("gt50_final_base_score", SOURCE_SCORER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def model_tokens(value: object) -> set[str]:
    return {token.lower() for token in MODEL_TOKEN_RE.findall(str(value or ""))}


def specific_designators_conflict(left: object, right: object) -> bool:
    left_tokens = model_tokens(left)
    right_tokens = model_tokens(right)
    return bool(left_tokens and right_tokens and left_tokens.isdisjoint(right_tokens))


def polyethylene_subtype(module, value: object) -> str:
    families = module.material_families(value)
    if "pe-g-mah" in families:
        return ""
    text = str(value or "")
    for subtype, pattern in POLYETHYLENE_SUBTYPE_PATTERNS:
        if pattern.search(text):
            return subtype
    return ""


def identity_qualifiers(value: object) -> set[str]:
    text = str(value or "")
    return {
        qualifier
        for qualifier, pattern in IDENTITY_QUALIFIER_PATTERNS
        if pattern.search(text)
    }


def strong_material_equal(module, left: object, right: object) -> tuple[bool, str]:
    if specific_designators_conflict(left, right):
        return False, "conflicting_material_designator"
    left_subtype = polyethylene_subtype(module, left)
    right_subtype = polyethylene_subtype(module, right)
    if left_subtype and right_subtype and left_subtype != right_subtype:
        return False, "polyethylene_subtype_mismatch"
    equal, rule = module.material_equal(left, right)
    if equal and rule == "conservative_containment":
        if identity_qualifiers(left) != identity_qualifiers(right):
            return False, "containment_chemical_qualifier_mismatch"
    return equal, rule


def scored_material_equal(module, gt_value: object, pred_value: object) -> tuple[bool, str]:
    equal, rule = strong_material_equal(module, gt_value, pred_value)
    if equal or rule != "polyethylene_subtype_mismatch":
        return equal, rule
    gt_subtype = polyethylene_subtype(module, gt_value)
    pred_subtype = polyethylene_subtype(module, pred_value)
    if gt_subtype == "pe" and pred_subtype and pred_subtype != "pe":
        return True, "prediction_specific_with_generic_PE_GT"
    return False, rule


def effective_polyethylene_subtype(module, row: dict) -> str:
    name_subtype = polyethylene_subtype(module, row.get(module.COL["entity_name"], ""))
    type_subtype = polyethylene_subtype(module, row.get(PE_TYPE_FIELD, ""))
    if name_subtype and name_subtype != "pe":
        return name_subtype
    if type_subtype:
        return type_subtype
    return name_subtype


def entity_name_rows_equal(module, gt_row: dict, pred_row: dict) -> tuple[bool, str]:
    equal, rule = scored_material_equal(
        module,
        gt_row.get(module.COL["entity_name"], ""),
        pred_row.get(module.COL["entity_name"], ""),
    )
    if equal or rule != "polyethylene_subtype_mismatch":
        return equal, rule
    gt_subtype = effective_polyethylene_subtype(module, gt_row)
    pred_subtype = effective_polyethylene_subtype(module, pred_row)
    if gt_subtype and gt_subtype == pred_subtype:
        return True, "polyethylene_subtype_from_PE_type"
    return False, rule


def normalize_parts(module, frame: pd.DataFrame, source: str, audit: list[dict]) -> pd.DataFrame:
    normalized = frame.copy()
    relation_col = module.COL["part_relation"]
    method_col = module.COL["mod_method"]
    for index, row in normalized.iterrows():
        raw_relation = module.clean(row.get(relation_col, "")).strip().lower()
        raw_method = module.clean(row.get(method_col, "")).strip().lower()
        relation = RELATION_ALIASES.get(raw_relation, raw_relation)
        method = METHOD_ALIASES.get(raw_method, raw_method)
        normalized.at[index, relation_col] = relation
        normalized.at[index, method_col] = method
        if relation != raw_relation or method != raw_method:
            audit.append(
                {
                    "source": source,
                    "raw_relation": raw_relation,
                    "canonical_relation": relation,
                    "raw_method": raw_method,
                    "canonical_method": method,
                }
            )
    return normalized


def scope_filter(
    module,
    frame: pd.DataFrame,
    view: str,
    source: str,
    audit: list[dict],
) -> pd.DataFrame:
    scoped = frame.copy()
    if view != "performance":
        return scoped

    sample_name = scoped[module.COL["sample_name"]].astype(str).str.lower()
    secondary_condition = sample_name.str.contains(
        SECONDARY_CONDITION_PATTERN,
        regex=True,
        na=False,
    )
    for _, row in scoped[secondary_condition].iterrows():
        audit.append(
            {
                "source": source,
                "view": view,
                "article": module.norm_article(row.get(module.COL["article"], "")),
                "sample_no": row.get(module.COL["sample_no"], ""),
                "sample_name": row.get(module.COL["sample_name"], ""),
                "reason": "secondary water-treatment or thermal-aging condition",
            }
        )
    return scoped[~secondary_condition].copy()


def load_method_dir(
    module,
    method_dir: Path,
    selected: set[str],
    source: str,
    scope_audit: list[dict],
) -> dict[str, pd.DataFrame]:
    files = {
        "entities": "formulation_entities.xlsx",
        "parts": "formulation_entity_parts.xlsx",
        "performance": "performance_samples_final.xlsx",
    }
    views: dict[str, pd.DataFrame] = {}
    for view, file_name in files.items():
        frame = module.read_xlsx(method_dir / file_name)
        frame[module.COL["article"]] = frame[module.COL["article"]].map(module.norm_article)
        frame = frame[frame[module.COL["article"]].isin(selected)].copy()
        views[view] = scope_filter(module, frame, view, source, scope_audit)
    return views


def load_methods(module, relation_audit: list[dict], scope_audit: list[dict]):
    selected_frame = module.read_xlsx(GT_PATH, "selected_articles")
    selected = {
        module.norm_article(value)
        for value in selected_frame["article_index"]
        if module.norm_article(value)
    }
    gt = {
        "entities": module.read_xlsx(GT_PATH, "formulation_entities_GT"),
        "parts": module.read_xlsx(GT_PATH, "entity_parts_GT"),
        "performance": module.read_xlsx(GT_PATH, "performance_GT"),
    }

    methods = {
        "Direct LLM": load_method_dir(
            module, DIRECT_DIR, selected, "Direct LLM", scope_audit
        ),
        "Expert Prompt Only": load_method_dir(
            module, EXPERT_DIR, selected, "Expert Prompt Only", scope_audit
        ),
        "Task Decomposition Only": load_method_dir(
            module, SPLIT_DIR, selected, "Task Decomposition Only", scope_audit
        ),
        "Full Pipeline": load_method_dir(
            module, FULL_DIR, selected, "Full Pipeline", scope_audit
        ),
    }

    if RERUN_ROOT is not None:
        split_replacements = {"13", "544"}
        for view, file_name in {
            "entities": "formulation_entities.xlsx",
            "parts": "formulation_entity_parts.xlsx",
            "performance": "performance_samples_final.xlsx",
        }.items():
            rerun = module.read_xlsx(RERUN_ROOT / "split_naive" / file_name)
            base = methods["Task Decomposition Only"][view].copy()
            rerun = scope_filter(
                module,
                rerun,
                view,
                "Task Decomposition Only rerun",
                scope_audit,
            )
            base[module.COL["article"]] = base[module.COL["article"]].map(module.norm_article)
            rerun[module.COL["article"]] = rerun[module.COL["article"]].map(module.norm_article)
            methods["Task Decomposition Only"][view] = pd.concat(
                [
                    base[~base[module.COL["article"]].isin(split_replacements)],
                    rerun[rerun[module.COL["article"]].isin(split_replacements)],
                ],
                ignore_index=True,
            ).fillna("")

    full_performance = methods["Full Pipeline"]["performance"]
    methods["Full Pipeline"]["performance"] = module.repair_full_performance_from_raw(
        full_performance,
        FULL_DIR,
    )

    gt["parts"] = normalize_parts(module, gt["parts"], "GT", relation_audit)
    for method, views in methods.items():
        views["parts"] = normalize_parts(module, views["parts"], method, relation_audit)

    for views in [gt, *methods.values()]:
        for frame in views.values():
            frame[module.COL["article"]] = frame[module.COL["article"]].map(module.norm_article)
            frame["_sample_uid"] = frame.apply(module.sample_uid, axis=1)
    return selected, gt, methods


def condition_signature(value: object) -> dict[str, tuple[Decimal, ...]]:
    text = str(value or "")
    signature: dict[str, tuple[Decimal, ...]] = {}
    for dimension, pattern in CONDITION_PATTERNS.items():
        values = tuple(Decimal(match) for match in pattern.findall(text))
        if values:
            signature[dimension] = values
    if "loading_wt" in signature:
        signature.pop("loading_pct", None)
    return signature


def values_overlap(left: tuple[Decimal, ...], right: tuple[Decimal, ...], tolerance=Decimal("0.05")) -> bool:
    return any(abs(left_value - right_value) <= tolerance for left_value in left for right_value in right)


def signature_conflict(left_name: object, right_name: object) -> tuple[bool, str]:
    left = condition_signature(left_name)
    right = condition_signature(right_name)
    for dimension in left.keys() & right.keys():
        if not values_overlap(left[dimension], right[dimension]):
            return True, f"conflicting_{dimension}"

    left_families = {
        "loading" if key.startswith("loading_") else "time" if key.startswith("time_") else key
        for key in left
    }
    right_families = {
        "loading" if key.startswith("loading_") else "time" if key.startswith("time_") else key
        for key in right
    }
    if left_families and right_families and left_families.isdisjoint(right_families):
        return True, "incompatible_condition_dimensions"
    return False, "compatible_condition_signature"


def matrix_presence(module, sample: dict) -> bool:
    return any(module.matrix_like(row) for row in sample["rows"])


def matrix_material_identities(module, rows: list[dict]) -> set[str]:
    identities: set[str] = set()
    for row in rows:
        if not module.matrix_like(row):
            continue
        name = row.get(module.COL["entity_name"], "")
        families = module.material_families(name)
        identities.update(families or {module.token_key(name)})
    return {identity for identity in identities if identity}


def positive_nonmatrix_presence(module, sample: dict) -> bool:
    for row in sample["rows"]:
        if module.matrix_like(row):
            continue
        loading = module.parse_loading(row.get("entity_loading_normalized_wt_or_phr", ""))
        if not loading:
            continue
        if any(
            (module.first_numeric(value) or Decimal("0")) > 0
            for value in loading.values()
        ):
            return True
    return False


def row_family(module, row: dict) -> tuple[str, ...]:
    name = row.get(module.COL["entity_name"], "")
    identities = tuple(sorted(material_identities(module, name)))
    return identities or (module.token_key(name),)


def shared_loading_conflict(module, left_rows: list[dict], right_rows: list[dict]) -> tuple[bool, str]:
    left_nonmatrix = [row for row in left_rows if not module.matrix_like(row)]
    right_nonmatrix = [row for row in right_rows if not module.matrix_like(row)]
    left_counts = Counter(row_family(module, row) for row in left_nonmatrix)
    right_counts = Counter(row_family(module, row) for row in right_nonmatrix)

    for left_row in left_nonmatrix:
        left_name = left_row.get(module.COL["entity_name"], "")
        left_family = row_family(module, left_row)
        for right_row in right_nonmatrix:
            right_name = right_row.get(module.COL["entity_name"], "")
            right_family = row_family(module, right_row)
            exact = module.token_key(left_name) == module.token_key(right_name)
            unique_family = (
                left_family == right_family
                and left_counts[left_family] == 1
                and right_counts[right_family] == 1
            )
            if not exact and not unique_family:
                continue
            left_loading = module.parse_loading(
                left_row.get("entity_loading_normalized_wt_or_phr", "")
            )
            right_loading = module.parse_loading(
                right_row.get("entity_loading_normalized_wt_or_phr", "")
            )
            common_units = left_loading.keys() & right_loading.keys()
            if common_units:
                unit_matches = {
                    unit: module.numeric_equal(
                        left_loading[unit],
                        right_loading[unit],
                        "entity_loading_normalized_wt_or_phr",
                    )[0]
                    for unit in common_units
                }
                if not any(unit_matches.values()):
                    units = "+".join(sorted(unit_matches))
                    return True, f"conflicting_shared_entity_loading:{module.clean(left_name)}:{units}"
    return False, "compatible_shared_entity_loadings"


GENERIC_MATERIAL_IDENTITIES = {
    "additive",
    "filler",
    "flame retardant",
    "main flame retardant",
    "synergist",
    "synergistic flame retardant",
}

BASE_MATERIAL_ABBREVIATIONS = {
    "APP",
    "ATH",
    "EG",
    "EVA",
    "HDPE",
    "IFR",
    "LDPE",
    "LLDPE",
    "MCA",
    "MDH",
    "MH",
    "MMT",
    "PE",
    "PER",
    "POE",
    "PP",
    "RP",
    "ZB",
}

DESIGNATOR_ALIASES = {
    "AMEO": "APTES",
    "KH550": "APTES",
}

CHEMICAL_IDENTITY_PATTERNS = [
    ("ceric-ammonium-nitrate", re.compile(r"(?i)\(nh4\)2\s*ce\s*\(no3\)6|ceric\s+ammonium\s+nitrate")),
    ("sodium-nitrate", re.compile(r"(?i)\bnano3\b|sodium\s+nitrate")),
    ("aluminum-nitrate", re.compile(r"(?i)al\s*\(no3\)3|alumin(?:um|ium)\s+nitrate")),
    ("iron-nitrate", re.compile(r"(?i)fe\s*\(no3\)3|ferric\s+nitrate|iron\s*\(?iii\)?\s+nitrate")),
    ("copper-nitrate", re.compile(r"(?i)cu\s*\(no3\)2|copper\s*\(?ii\)?\s+nitrate")),
    ("zinc-nitrate", re.compile(r"(?i)zn\s*\(no3\)2|zinc\s+nitrate")),
    ("nitric-acid", re.compile(r"(?i)\bhno3\b|nitric\s+acid")),
    ("zinc-sulfate", re.compile(r"(?i)zn\s*so4|zinc\s+sulfate")),
    ("iron-sulfate", re.compile(r"(?i)fe2\s*\(so4\)3|ferric\s+sulfate|iron\s*\(?iii\)?\s+sulfate")),
]

MATERIAL_IDENTITY_PATTERNS = [
    ("ath", re.compile(r"(?i)alumin(?:um|ium)\s+(?:tri)?hydroxide|alumina\s+trihydrate|aluminum\s+trihydrate")),
]


def chemical_material_identities(value: object) -> set[str]:
    text = str(value or "")
    return {identity for identity, pattern in CHEMICAL_IDENTITY_PATTERNS if pattern.search(text)}


def material_identities(module, value: object) -> set[str]:
    chemical = chemical_material_identities(value)
    if chemical:
        return chemical
    text = str(value or "")
    aliases = {identity for identity, pattern in MATERIAL_IDENTITY_PATTERNS if pattern.search(text)}
    if aliases:
        return aliases
    families = module.material_families(value)
    if families:
        return families
    token = module.token_key(value)
    return {token} if token and token not in GENERIC_MATERIAL_IDENTITIES else set()


def specific_material_designators(value: object) -> set[str]:
    text = str(value or "")
    tokens = set(re.findall(r"\b[A-Z]{2,10}[A-Z0-9.-]*\b", text))
    tokens.update(
        token.upper()
        for token in MODEL_TOKEN_RE.findall(text)
        if any(character.isalpha() for character in token)
        and any(character.isdigit() for character in token)
    )
    normalized = {DESIGNATOR_ALIASES.get(token, token) for token in tokens}
    return normalized - BASE_MATERIAL_ABBREVIATIONS


def nonmatrix_material_identities(module, rows: list[dict]) -> set[str]:
    identities: set[str] = set()
    for row in rows:
        if module.matrix_like(row):
            continue
        identities.update(material_identities(module, row.get(module.COL["entity_name"], "")))
    return identities


def bidirectional_material_conflict(module, left_rows: list[dict], right_rows: list[dict]) -> tuple[bool, str]:
    left_identities = nonmatrix_material_identities(module, left_rows)
    right_identities = nonmatrix_material_identities(module, right_rows)
    if "ifr" in left_identities or "ifr" in right_identities:
        return False, "package_representation_allowed"
    shared = left_identities & right_identities
    left_only = left_identities - right_identities
    right_only = right_identities - left_identities
    if shared and left_only and right_only:
        detail = f"{'+'.join(sorted(left_only))}_vs_{'+'.join(sorted(right_only))}"
        return True, f"conflicting_nonmatrix_material_families:{detail}"
    return False, "compatible_nonmatrix_material_families"


def specific_designator_conflict(module, left_rows: list[dict], right_rows: list[dict]) -> tuple[bool, str]:
    for left_row in left_rows:
        if module.matrix_like(left_row):
            continue
        left_name = left_row.get(module.COL["entity_name"], "")
        left_families = module.material_families(left_name)
        left_designators = specific_material_designators(left_name)
        if not left_families or not left_designators:
            continue
        for right_row in right_rows:
            if module.matrix_like(right_row):
                continue
            right_name = right_row.get(module.COL["entity_name"], "")
            right_families = module.material_families(right_name)
            right_designators = specific_material_designators(right_name)
            if left_families.isdisjoint(right_families) or not right_designators:
                continue
            if left_designators.isdisjoint(right_designators):
                detail = f"{'+'.join(sorted(left_designators))}_vs_{'+'.join(sorted(right_designators))}"
                return True, f"conflicting_specific_material_designators:{detail}"
    return False, "compatible_specific_material_designators"


def sample_compatible(module, left: dict, right: dict) -> tuple[bool, str]:
    left_label = module.token_key(left["sample_name"])
    right_label = module.token_key(right["sample_name"])
    if left_label and left_label == right_label:
        return True, "exact_sample_identifier"
    if matrix_presence(module, left) != matrix_presence(module, right):
        return False, "matrix_presence_mismatch"
    left_matrices = matrix_material_identities(module, left["rows"])
    right_matrices = matrix_material_identities(module, right["rows"])
    if left_matrices and right_matrices and left_matrices.isdisjoint(right_matrices):
        detail = f"{'+'.join(sorted(left_matrices))}_vs_{'+'.join(sorted(right_matrices))}"
        return False, f"conflicting_matrix_material_families:{detail}"
    conflict, reason = specific_designator_conflict(module, left["rows"], right["rows"])
    if conflict:
        return False, reason
    conflict, reason = bidirectional_material_conflict(module, left["rows"], right["rows"])
    if conflict:
        return False, reason
    conflict, reason = shared_loading_conflict(module, left["rows"], right["rows"])
    if conflict:
        return False, reason
    conflict, reason = signature_conflict(left["sample_name"], right["sample_name"])
    if conflict:
        return False, reason
    left_signature = condition_signature(left["sample_name"])
    right_signature = condition_signature(right["sample_name"])
    exact_label = module.token_key(left["sample_name"]) == module.token_key(right["sample_name"])
    if (
        bool(left_signature) != bool(right_signature)
        and positive_nonmatrix_presence(module, left) != positive_nonmatrix_presence(module, right)
        and not exact_label
    ):
        return False, "explicit_condition_with_nonmatrix_presence_mismatch"
    return True, "compatible"


def condition_alignment_score(left_name: object, right_name: object) -> float:
    left = condition_signature(left_name)
    right = condition_signature(right_name)
    shared = left.keys() & right.keys()
    if not shared:
        return 0.0
    if all(values_overlap(left[dimension], right[dimension]) for dimension in shared):
        left_families = {
            "loading" if key.startswith("loading_") else "time" if key.startswith("time_") else key
            for key in left
        }
        right_families = {
            "loading" if key.startswith("loading_") else "time" if key.startswith("time_") else key
            for key in right
        }
        if left_families == right_families:
            return 0.90
        return 0.72
    return 0.0


def nonmatrix_loading_totals(module, rows: list[dict]) -> dict[str, Decimal]:
    totals: dict[str, Decimal] = defaultdict(Decimal)
    for row in rows:
        if module.matrix_like(row):
            continue
        for unit, value in module.parse_loading(
            row.get("entity_loading_normalized_wt_or_phr", "")
        ).items():
            numeric = module.first_numeric(value)
            if numeric is not None:
                totals[unit] += numeric
    return dict(totals)


def package_alignment_score(module, left: dict, right: dict) -> float:
    left_ids = nonmatrix_material_identities(module, left["rows"])
    right_ids = nonmatrix_material_identities(module, right["rows"])
    if ("ifr" in left_ids) == ("ifr" in right_ids):
        return 0.0
    left_matrices = matrix_material_identities(module, left["rows"])
    right_matrices = matrix_material_identities(module, right["rows"])
    if left_matrices and right_matrices and left_matrices.isdisjoint(right_matrices):
        return 0.0
    left_totals = nonmatrix_loading_totals(module, left["rows"])
    right_totals = nonmatrix_loading_totals(module, right["rows"])
    for unit in left_totals.keys() & right_totals.keys():
        if module.numeric_equal(
            left_totals[unit],
            right_totals[unit],
            "entity_loading_normalized_wt_or_phr",
        )[0]:
            return 0.70
    return 0.0


def build_sample_alignment(module, rejection_rows: list[dict]):
    def align(selected, gt_entities, methods, overrides):
        del overrides
        records: list[dict] = []
        mappings: dict[str, dict[str, dict[str, str]]] = {}
        for method, views in methods.items():
            mappings[method] = {}
            pred_entities = views["entities"]
            for article in sorted(selected, key=lambda value: int(value) if value.isdigit() else value):
                gt_samples = module.sample_records(gt_entities, article)
                pred_samples = module.sample_records(pred_entities, article)
                score_lookup: dict[tuple[str, str], float] = {}
                composition_lookup: dict[tuple[str, str], float] = {}
                compatibility_lookup: dict[tuple[str, str], tuple[bool, str]] = {}
                for gt_sample in gt_samples:
                    for pred_sample in pred_samples:
                        compatible, reason = sample_compatible(module, gt_sample, pred_sample)
                        compatibility_lookup[(gt_sample["uid"], pred_sample["uid"])] = (compatible, reason)
                        if compatible:
                            composition_score = module.sample_pair_score(gt_sample, pred_sample)
                            condition_score = condition_alignment_score(
                                gt_sample["sample_name"], pred_sample["sample_name"]
                            )
                            package_score = package_alignment_score(module, gt_sample, pred_sample)
                            score = max(composition_score, condition_score, package_score)
                        else:
                            composition_score = 0.0
                            score = 0.0
                        composition_lookup[(gt_sample["uid"], pred_sample["uid"])] = composition_score
                        score_lookup[(gt_sample["uid"], pred_sample["uid"])] = score
                        if not compatible and score == 0.0:
                            rejection_rows.append(
                                {
                                    "method": method,
                                    "article": article,
                                    "gt_sample_no": gt_sample["sample_no"],
                                    "gt_sample_name": gt_sample["sample_name"],
                                    "pred_sample_no": pred_sample["sample_no"],
                                    "pred_sample_name": pred_sample["sample_name"],
                                    "rejection_reason": reason,
                                }
                            )

                gt_labels = defaultdict(list)
                pred_labels = defaultdict(list)
                for index, sample in enumerate(gt_samples):
                    gt_labels[module.token_key(sample["sample_name"])].append(index)
                for index, sample in enumerate(pred_samples):
                    pred_labels[module.token_key(sample["sample_name"])].append(index)

                identifier_pairs = []
                for label, gt_indexes in gt_labels.items():
                    pred_indexes = pred_labels.get(label, [])
                    if label and len(gt_indexes) == 1 and len(pred_indexes) == 1:
                        gt_index = gt_indexes[0]
                        pred_index = pred_indexes[0]
                        gt_sample = gt_samples[gt_index]
                        pred_sample = pred_samples[pred_index]
                        compatible, _ = compatibility_lookup[(gt_sample["uid"], pred_sample["uid"])]
                        if compatible:
                            identifier_pairs.append(
                                (
                                    gt_index,
                                    pred_index,
                                    score_lookup[(gt_sample["uid"], pred_sample["uid"])],
                                )
                            )

                identifier_gt = {left for left, _, _ in identifier_pairs}
                identifier_pred = {right for _, right, _ in identifier_pairs}
                high_confidence_pairs = []
                candidate_gt_indexes = [i for i in range(len(gt_samples)) if i not in identifier_gt]
                candidate_pred_indexes = [i for i in range(len(pred_samples)) if i not in identifier_pred]
                gt_best: dict[int, tuple[int, float]] = {}
                pred_best: dict[int, tuple[int, float]] = {}
                for gt_index in candidate_gt_indexes:
                    scores = [
                        (
                            pred_index,
                            composition_lookup[(gt_samples[gt_index]["uid"], pred_samples[pred_index]["uid"])],
                        )
                        for pred_index in candidate_pred_indexes
                    ]
                    if scores:
                        best_score = max(score for _, score in scores)
                        best = [pred_index for pred_index, score in scores if abs(score - best_score) < 1e-12]
                        if best_score >= 0.82 and len(best) == 1:
                            gt_best[gt_index] = (best[0], best_score)
                for pred_index in candidate_pred_indexes:
                    scores = [
                        (
                            gt_index,
                            composition_lookup[(gt_samples[gt_index]["uid"], pred_samples[pred_index]["uid"])],
                        )
                        for gt_index in candidate_gt_indexes
                    ]
                    if scores:
                        best_score = max(score for _, score in scores)
                        best = [gt_index for gt_index, score in scores if abs(score - best_score) < 1e-12]
                        if best_score >= 0.82 and len(best) == 1:
                            pred_best[pred_index] = (best[0], best_score)
                for gt_index, (pred_index, score) in gt_best.items():
                    if pred_best.get(pred_index, (None, 0.0))[0] == gt_index:
                        high_confidence_pairs.append((gt_index, pred_index, score))

                anchored_gt = identifier_gt | {left for left, _, _ in high_confidence_pairs}
                anchored_pred = identifier_pred | {right for _, right, _ in high_confidence_pairs}
                remaining_gt_indexes = [i for i in range(len(gt_samples)) if i not in anchored_gt]
                remaining_pred_indexes = [i for i in range(len(pred_samples)) if i not in anchored_pred]
                remaining_gt = [gt_samples[i] for i in remaining_gt_indexes]
                remaining_pred = [pred_samples[i] for i in remaining_pred_indexes]
                score_function = lambda left, right: score_lookup[(left["uid"], right["uid"])]
                remaining_pairs, _, _ = module.assign_pairs(
                    remaining_gt,
                    remaining_pred,
                    score_function,
                    0.28,
                )
                filtered_remaining_pairs = []
                for left, right, score in remaining_pairs:
                    gt_sample = remaining_gt[left]
                    pred_sample = remaining_pred[right]
                    candidate_scores = sorted(
                        [
                            score_lookup[(gt_sample["uid"], candidate["uid"])]
                            for candidate in pred_samples
                        ],
                        reverse=True,
                    )
                    margin = score - candidate_scores[1] if len(candidate_scores) > 1 else score
                    if score < 0.55 and margin <= 0:
                        rejection_rows.append(
                            {
                                "method": method,
                                "article": article,
                                "gt_sample_no": gt_sample["sample_no"],
                                "gt_sample_name": gt_sample["sample_name"],
                                "pred_sample_no": pred_sample["sample_no"],
                                "pred_sample_name": pred_sample["sample_name"],
                                "rejection_reason": "low_score_nonpositive_margin",
                            }
                        )
                        continue
                    filtered_remaining_pairs.append((left, right, score))
                remaining_pairs = filtered_remaining_pairs
                composition_pairs = [
                    (remaining_gt_indexes[left], remaining_pred_indexes[right], score)
                    for left, right, score in remaining_pairs
                ]
                pairs = identifier_pairs + high_confidence_pairs + composition_pairs
                used_gt = {left for left, _, _ in pairs}
                used_pred = {right for _, right, _ in pairs}
                unmatched_gt = [i for i in range(len(gt_samples)) if i not in used_gt]
                unmatched_pred = [i for i in range(len(pred_samples)) if i not in used_pred]

                article_mapping: dict[str, str] = {}
                for gt_index, pred_index, score in pairs:
                    gt_sample = gt_samples[gt_index]
                    pred_sample = pred_samples[pred_index]
                    compatible_scores = sorted(
                        [
                            score_lookup[(gt_sample["uid"], candidate["uid"])]
                            for candidate in pred_samples
                        ],
                        reverse=True,
                    )
                    margin = score - compatible_scores[1] if len(compatible_scores) > 1 else score
                    gt_label = module.token_key(gt_sample["sample_name"])
                    pred_label = module.token_key(pred_sample["sample_name"])
                    exact_label = bool(gt_label) and gt_label == pred_label
                    review_priority = (
                        "low"
                        if score >= 0.82 and margin >= 0.10
                        else "medium"
                        if score >= 0.62 and margin >= 0.05
                        else "high"
                    )
                    article_mapping[gt_sample["uid"]] = pred_sample["uid"]
                    records.append(
                        {
                            "method": method,
                            "article": article,
                            "status": "paired",
                            "review_priority": review_priority,
                            "gt_uid": gt_sample["uid"],
                            "pred_uid": pred_sample["uid"],
                            "gt_sample_no": gt_sample["sample_no"],
                            "pred_sample_no": pred_sample["sample_no"],
                            "gt_sample_name": gt_sample["sample_name"],
                            "pred_sample_name": pred_sample["sample_name"],
                            "formulation_score": round(score, 6),
                            "next_best_margin": round(margin, 6),
                            "label_exact": exact_label,
                            "gt_formulation": module.formulation_summary(gt_sample["rows"]),
                            "pred_formulation": module.formulation_summary(pred_sample["rows"]),
                            "decision_source": (
                                "exact_sample_identifier"
                                if (gt_index, pred_index, score) in identifier_pairs
                                else "mutual_high_confidence_formulation"
                                if (gt_index, pred_index, score) in high_confidence_pairs
                                else "composition_with_conflict_constraints"
                            ),
                            "adjudication_note": "",
                        }
                    )

                for index in unmatched_gt:
                    sample = gt_samples[index]
                    records.append(
                        {
                            "method": method,
                            "article": article,
                            "status": "gt_only",
                            "review_priority": "high",
                            "gt_uid": sample["uid"],
                            "pred_uid": "",
                            "gt_sample_no": sample["sample_no"],
                            "pred_sample_no": "",
                            "gt_sample_name": sample["sample_name"],
                            "pred_sample_name": "",
                            "formulation_score": 0,
                            "next_best_margin": 0,
                            "label_exact": False,
                            "gt_formulation": module.formulation_summary(sample["rows"]),
                            "pred_formulation": "",
                            "decision_source": "composition_with_conflict_constraints",
                            "adjudication_note": "",
                        }
                    )
                for index in unmatched_pred:
                    sample = pred_samples[index]
                    records.append(
                        {
                            "method": method,
                            "article": article,
                            "status": "pred_only",
                            "review_priority": "high",
                            "gt_uid": "",
                            "pred_uid": sample["uid"],
                            "gt_sample_no": "",
                            "pred_sample_no": sample["sample_no"],
                            "gt_sample_name": "",
                            "pred_sample_name": sample["sample_name"],
                            "formulation_score": 0,
                            "next_best_margin": 0,
                            "label_exact": False,
                            "gt_formulation": "",
                            "pred_formulation": module.formulation_summary(sample["rows"]),
                            "decision_source": "composition_with_conflict_constraints",
                            "adjudication_note": "",
                        }
                    )
                mappings[method][article] = article_mapping
        return pd.DataFrame(records), mappings

    return align


def augment_loadings_with_provenance(module, frame: pd.DataFrame) -> pd.DataFrame:
    augmented = frame.copy()
    augmented[NORMALIZED_LOADING_FIELD] = ""
    augmented[LOADING_WT_ORIGIN] = ""
    augmented[LOADING_PHR_ORIGIN] = ""
    for _, indexes in augmented.groupby("_sample_uid", sort=False).groups.items():
        rows = {index: augmented.loc[index].to_dict() for index in indexes}
        matrix_indexes = [index for index in indexes if module.matrix_like(rows[index])]
        nonmatrix_indexes = [index for index in indexes if index not in matrix_indexes]
        wt_values = {
            index: module.first_numeric(rows[index].get(module.COL["entity_amount_wt"], ""))
            for index in indexes
        }
        phr_values = {
            index: module.first_numeric(rows[index].get(module.COL["entity_amount_phr"], ""))
            for index in indexes
        }
        matrix_wt = sum(
            (wt_values[index] for index in matrix_indexes if wt_values[index] is not None),
            Decimal("0"),
        )
        matrix_wt = matrix_wt if matrix_wt > 0 else None
        all_nonmatrix_phr_known = all(
            phr_values[index] is not None for index in nonmatrix_indexes
        )
        nonmatrix_phr = [
            phr_values[index]
            for index in nonmatrix_indexes
            if phr_values[index] is not None
        ]
        phr_denominator = (
            Decimal("100") + sum(nonmatrix_phr, Decimal("0"))
            if matrix_indexes and all_nonmatrix_phr_known
            else None
        )

        for index in indexes:
            wt_value = wt_values[index]
            phr_value = phr_values[index]
            wt_origin = "explicit" if wt_value is not None else ""
            phr_origin = "explicit" if phr_value is not None else ""
            if wt_value is None and phr_value is not None:
                if matrix_wt is not None and index not in matrix_indexes:
                    wt_value = phr_value * matrix_wt / Decimal("100")
                    wt_origin = "derived_from_phr_and_matrix_wt"
                elif phr_denominator is not None:
                    wt_value = phr_value / phr_denominator * Decimal("100")
                    wt_origin = "derived_from_complete_phr_denominator"
            if phr_value is None and wt_value is not None and matrix_wt is not None:
                phr_value = wt_value / matrix_wt * Decimal("100")
                phr_origin = "derived_from_wt_and_matrix_wt"

            values = []
            if wt_value is not None:
                values.append(f"wt={module.decimal_text(wt_value)}")
            if phr_value is not None:
                values.append(f"phr={module.decimal_text(phr_value)}")
            augmented.at[index, NORMALIZED_LOADING_FIELD] = ";".join(values)
            augmented.at[index, LOADING_WT_ORIGIN] = wt_origin
            augmented.at[index, LOADING_PHR_ORIGIN] = phr_origin
    return augmented


def loading_rows_equal(module, gt_row: dict, pred_row: dict) -> tuple[bool, str]:
    gt_values = module.parse_loading(gt_row.get(NORMALIZED_LOADING_FIELD, ""))
    pred_values = module.parse_loading(pred_row.get(NORMALIZED_LOADING_FIELD, ""))
    origin_columns = {
        "wt": LOADING_WT_ORIGIN,
        "phr": LOADING_PHR_ORIGIN,
    }

    for unit in ("wt", "phr"):
        if unit not in gt_values or unit not in pred_values:
            continue
        gt_origin = gt_row.get(origin_columns[unit], "")
        pred_origin = pred_row.get(origin_columns[unit], "")
        if gt_origin == "explicit" and pred_origin == "explicit":
            equal, _ = strict_numeric_equal(
                module,
                gt_values[unit],
                pred_values[unit],
                NORMALIZED_LOADING_FIELD,
            )
            if equal:
                return True, f"loading_{unit}_explicit"

    for unit in ("wt", "phr"):
        if unit not in gt_values or unit not in pred_values:
            continue
        gt_origin = gt_row.get(origin_columns[unit], "")
        pred_origin = pred_row.get(origin_columns[unit], "")
        cross_basis = (
            gt_origin == "explicit" and pred_origin.startswith("derived_")
        ) or (
            pred_origin == "explicit" and gt_origin.startswith("derived_")
        )
        if not cross_basis:
            continue
        equal, _ = strict_numeric_equal(
            module,
            gt_values[unit],
            pred_values[unit],
            NORMALIZED_LOADING_FIELD,
        )
        if equal:
            return True, f"loading_{unit}_explicit_to_derived"
    return False, "loading_mismatch"


def strict_numeric_equal(
    module,
    left,
    right,
    field: str,
    article: str = "",
) -> tuple[bool, str]:
    left_values = module.numeric_values(left)
    right_values = module.numeric_values(right)
    if not left_values and not right_values:
        return True, "both_blank"
    if not left_values or not right_values:
        return False, "one_blank"

    left_text = module.clean(left)
    right_text = module.clean(right)
    if field in module.PERFORMANCE_FIELDS and re.match(r"^\s*[<>]", left_text):
        bound = left_values[0]
        candidate = right_values[0]
        if (left_text.lstrip().startswith(">") and candidate > bound) or (
            left_text.lstrip().startswith("<") and candidate < bound
        ):
            return True, "GT_reported_bound_satisfied"

    uncertainty_re = re.compile('\xb1|\\+/-|\\\\pm', re.IGNORECASE)
    if uncertainty_re.search(left_text):
        left_values = left_values[:1]
    if uncertainty_re.search(right_text):
        right_values = right_values[:1]

    if len(left_values) != len(right_values):
        return False, "numeric_arity_mismatch"

    scale_factors = VERIFIED_SCALE_FACTORS.get((module.norm_article(article), field), ())

    rules = []
    for left_value, right_value in zip(left_values, right_values):
        if module.numeric_close(left_value, right_value, field):
            rules.append("numeric_tolerance")
            continue
        converted = False
        for factor in scale_factors:
            if module.numeric_close(left_value * factor, right_value, field) or module.numeric_close(
                left_value,
                right_value * factor,
                field,
            ):
                rules.append(f"documented_scale_x{factor}")
                converted = True
                break
        if not converted:
            return False, "numeric_mismatch"
    return True, "+".join(sorted(set(rules)))


def strict_field_equal(module, original_field_equal):
    def compare(field: str, left, right) -> tuple[bool, str]:
        if field in {module.COL["entity_name"], module.COL["part_name"]}:
            return scored_material_equal(module, module.clean(left), module.clean(right))
        if field == module.COL["part_amount"] or field in module.PERFORMANCE_FIELDS:
            return strict_numeric_equal(module, left, right, field)
        return original_field_equal(field, left, right)

    return compare


def strict_score_pair(module):
    def score(method, view, article, gt_row, pred_row, fields, context):
        records = []
        for field in fields:
            gt_value = module.clean(gt_row.get(field, ""))
            pred_value = module.clean(pred_row.get(field, ""))
            if view == "performance" and module.VISUAL_RE.match(gt_value):
                continue
            if view == "performance" and module.VISUAL_RE.match(pred_value):
                pred_value = ""
            gt_present = not module.is_missing(gt_value)
            pred_present = not module.is_missing(pred_value)
            if not gt_present and not pred_present:
                continue
            if gt_present and pred_present:
                if view == "entities" and field == module.COL["entity_name"]:
                    equal, rule = entity_name_rows_equal(module, gt_row, pred_row)
                elif view == "entities" and field == NORMALIZED_LOADING_FIELD:
                    equal, rule = loading_rows_equal(module, gt_row, pred_row)
                elif field == module.COL["part_amount"] or (
                    field in module.PERFORMANCE_FIELDS and field != UL94_FIELD
                ):
                    equal, rule = strict_numeric_equal(
                        module,
                        gt_value,
                        pred_value,
                        field,
                        article,
                    )
                else:
                    equal, rule = module.field_equal(field, gt_value, pred_value)
                outcome = "TP" if equal else "FP+FN"
            elif gt_present:
                rule = "prediction_missing"
                outcome = "FN"
            else:
                rule = "prediction_extra"
                outcome = "FP"
            records.append(
                {
                    "method": method,
                    "article": article,
                    "view": view,
                    **context,
                    "field": field,
                    "GT": gt_value,
                    "PRED": pred_value,
                    "outcome": outcome,
                    "equivalence_rule": rule,
                    "adjudication_note": "",
                }
            )
        return records

    return score


def patched_entity_pair_score(module, original):
    def score(left: dict, right: dict) -> float:
        if specific_designators_conflict(
            left.get(module.COL["entity_name"], ""),
            right.get(module.COL["entity_name"], ""),
        ):
            return 0.0
        return original(left, right)

    return score


def patched_part_pair_score(module):
    def score(left: dict, right: dict) -> float:
        left_name = left.get(module.COL["part_name"], "")
        right_name = right.get(module.COL["part_name"], "")
        if specific_designators_conflict(left_name, right_name):
            return 0.0
        name_score = module.text_similarity(left_name, right_name)
        identity_equal, _ = strong_material_equal(module, left_name, right_name)
        if not identity_equal and name_score < 0.70:
            return 0.0
        role_score = float(
            module.canonical_category(module.COL["part_role"], left.get(module.COL["part_role"], ""))
            == module.canonical_category(module.COL["part_role"], right.get(module.COL["part_role"], ""))
        )
        relation_score = float(
            module.canonical_category(
                module.COL["part_relation"], left.get(module.COL["part_relation"], "")
            )
            == module.canonical_category(
                module.COL["part_relation"], right.get(module.COL["part_relation"], "")
            )
        )
        method_score = float(
            module.canonical_category(module.COL["mod_method"], left.get(module.COL["mod_method"], ""))
            == module.canonical_category(module.COL["mod_method"], right.get(module.COL["mod_method"], ""))
        )
        ratio_equal, _ = module.field_equal(
            module.COL["part_ratio"],
            left.get(module.COL["part_ratio"], ""),
            right.get(module.COL["part_ratio"], ""),
        )
        return 0.58 * name_score + 0.14 * role_score + 0.12 * relation_score + 0.08 * method_score + 0.08 * float(ratio_equal)

    return score


def dedupe_entity_rows(module):
    def dedupe(rows: list[dict]):
        merged_rows = []
        for row in rows:
            matched = None
            for candidate in merged_rows:
                same_material, _ = strong_material_equal(
                    module,
                    row.get(module.COL["entity_name"], ""),
                    candidate.get(module.COL["entity_name"], ""),
                )
                row_grade = module.token_key(row.get(module.COL["entity_grade"], ""))
                candidate_grade = module.token_key(candidate.get(module.COL["entity_grade"], ""))
                grades_compatible = not row_grade or not candidate_grade or row_grade == candidate_grade
                same_role = module.canonical_category(
                    module.COL["entity_role"], row.get(module.COL["entity_role"], "")
                ) == module.canonical_category(
                    module.COL["entity_role"], candidate.get(module.COL["entity_role"], "")
                )
                row_loading = row.get("entity_loading_normalized_wt_or_phr", "")
                candidate_loading = candidate.get("entity_loading_normalized_wt_or_phr", "")
                compatible_loading = (
                    module.is_missing(row_loading)
                    or module.is_missing(candidate_loading)
                    or module.loading_similarity(row_loading, candidate_loading) == 1.0
                )
                if same_material and grades_compatible and same_role and compatible_loading:
                    matched = candidate
                    break
            if matched is None:
                copied = dict(row)
                copied["_source_entity_uids"] = [row["_entity_uid"]]
                merged_rows.append(copied)
            else:
                module.merge_complementary_rows(matched, row, list(row))
                matched["_source_entity_uids"].append(row["_entity_uid"])
        return merged_rows

    return dedupe


def dedupe_part_rows(module):
    def dedupe(rows: list[dict]):
        merged_rows = []
        for row in rows:
            matched = None
            for candidate in merged_rows:
                same_material, _ = strong_material_equal(
                    module,
                    row.get(module.COL["part_name"], ""),
                    candidate.get(module.COL["part_name"], ""),
                )
                same_role = module.canonical_category(
                    module.COL["part_role"], row.get(module.COL["part_role"], "")
                ) == module.canonical_category(
                    module.COL["part_role"], candidate.get(module.COL["part_role"], "")
                )
                same_relation = module.canonical_category(
                    module.COL["part_relation"], row.get(module.COL["part_relation"], "")
                ) == module.canonical_category(
                    module.COL["part_relation"], candidate.get(module.COL["part_relation"], "")
                )
                row_ratio = row.get(module.COL["part_ratio"], "")
                candidate_ratio = candidate.get(module.COL["part_ratio"], "")
                ratio_compatible = (
                    module.is_missing(row_ratio)
                    or module.is_missing(candidate_ratio)
                    or module.field_equal(module.COL["part_ratio"], row_ratio, candidate_ratio)[0]
                )
                if same_material and same_role and same_relation and ratio_compatible:
                    matched = candidate
                    break
            if matched is None:
                copied = dict(row)
                copied["_source_part_uids"] = [row["_part_uid"]]
                merged_rows.append(copied)
            else:
                module.merge_complementary_rows(matched, row, list(row))
                matched["_source_part_uids"].append(row["_part_uid"])
        return merged_rows

    return dedupe


def performance_row_score(module, left: dict, right: dict) -> float:
    exact_name = module.token_key(left.get(module.COL["sample_name"], "")) == module.token_key(
        right.get(module.COL["sample_name"], "")
    )
    compared = 0
    matched = 0
    for field in module.PERFORMANCE_FIELDS:
        left_value = module.clean(left.get(field, ""))
        right_value = module.clean(right.get(field, ""))
        if module.is_missing(left_value) or module.is_missing(right_value):
            continue
        if module.VISUAL_RE.match(left_value) or module.VISUAL_RE.match(right_value):
            continue
        compared += 1
        article = module.norm_article(left.get(module.COL["article"], ""))
        if field == UL94_FIELD:
            equal = module.field_equal(field, left_value, right_value)[0]
        else:
            equal = strict_numeric_equal(module, left_value, right_value, field, article)[0]
        if equal:
            matched += 1
    value_score = matched / compared if compared else 0.0
    return 0.55 + 0.25 * float(exact_name) + 0.20 * value_score


def score_performance(module):
    def score(gt_perf, methods, sample_mappings, sample_aliases):
        records = []
        for method, views in methods.items():
            pred_perf = views["performance"].copy()
            used_gt = set()
            used_pred = set()
            for article, mapping in sample_mappings[method].items():
                for gt_sample, pred_sample in mapping.items():
                    gt_rows = gt_perf[gt_perf["_linked_sample_uid"] == gt_sample].to_dict("records")
                    pred_sample_ids = [pred_sample] + sample_aliases[method][article].get(gt_sample, [])
                    pred_rows = []
                    for pred_sample_id in pred_sample_ids:
                        pred_rows.extend(
                            pred_perf[pred_perf["_linked_sample_uid"] == pred_sample_id].to_dict("records")
                        )
                    pairs, unmatched_gt, unmatched_pred = module.assign_pairs(
                        gt_rows,
                        pred_rows,
                        lambda left, right: performance_row_score(module, left, right),
                        0.0,
                    )
                    for gt_index, pred_index, _ in pairs:
                        gt_row = gt_rows[gt_index]
                        pred_row = pred_rows[pred_index]
                        used_gt.add(gt_row["_performance_uid"])
                        used_pred.add(pred_row["_performance_uid"])
                        context = {
                            "gt_sample_uid": gt_sample,
                            "pred_sample_uid": pred_sample,
                            "gt_object_uid": gt_row["_performance_uid"],
                            "pred_object_uid": pred_row["_performance_uid"],
                        }
                        records.extend(
                            module.score_pair(
                                method,
                                "performance",
                                article,
                                gt_row,
                                pred_row,
                                module.PERFORMANCE_FIELDS,
                                context,
                            )
                        )
                    for index in unmatched_gt:
                        gt_row = gt_rows[index]
                        used_gt.add(gt_row["_performance_uid"])
                        context = {
                            "gt_sample_uid": gt_sample,
                            "pred_sample_uid": pred_sample,
                            "gt_object_uid": gt_row["_performance_uid"],
                            "pred_object_uid": "",
                        }
                        records.extend(
                            module.score_unmatched(
                                method,
                                "performance",
                                article,
                                gt_row,
                                module.PERFORMANCE_FIELDS,
                                "gt",
                                context,
                            )
                        )
                    for index in unmatched_pred:
                        pred_row = pred_rows[index]
                        used_pred.add(pred_row["_performance_uid"])
                        context = {
                            "gt_sample_uid": gt_sample,
                            "pred_sample_uid": pred_sample,
                            "gt_object_uid": "",
                            "pred_object_uid": pred_row["_performance_uid"],
                        }
                        records.extend(
                            module.score_unmatched(
                                method,
                                "performance",
                                article,
                                pred_row,
                                module.PERFORMANCE_FIELDS,
                                "pred",
                                context,
                            )
                        )
            for gt_row in gt_perf[~gt_perf["_performance_uid"].isin(used_gt)].to_dict("records"):
                context = {
                    "gt_sample_uid": gt_row.get("_linked_sample_uid", ""),
                    "pred_sample_uid": "",
                    "gt_object_uid": gt_row["_performance_uid"],
                    "pred_object_uid": "",
                }
                records.extend(
                    module.score_unmatched(
                        method,
                        "performance",
                        module.norm_article(gt_row.get(module.COL["article"], "")),
                        gt_row,
                        module.PERFORMANCE_FIELDS,
                        "gt",
                        context,
                    )
                )
            for pred_row in pred_perf[~pred_perf["_performance_uid"].isin(used_pred)].to_dict("records"):
                context = {
                    "gt_sample_uid": "",
                    "pred_sample_uid": pred_row.get("_linked_sample_uid", ""),
                    "gt_object_uid": "",
                    "pred_object_uid": pred_row["_performance_uid"],
                }
                records.extend(
                    module.score_unmatched(
                        method,
                        "performance",
                        module.norm_article(pred_row.get(module.COL["article"], "")),
                        pred_row,
                        module.PERFORMANCE_FIELDS,
                        "pred",
                        context,
                    )
                )
        return records

    return score


def write_relation_audit(rows: list[dict]) -> None:
    counts = Counter(
        (
            row["source"],
            row["raw_relation"],
            row["canonical_relation"],
            row["raw_method"],
            row["canonical_method"],
        )
        for row in rows
    )
    frame = pd.DataFrame(
        [
            {
                "source": key[0],
                "raw_relation": key[1],
                "canonical_relation": key[2],
                "raw_method": key[3],
                "canonical_method": key[4],
                "rows": count,
            }
            for key, count in counts.items()
        ]
    )
    frame.to_csv(
        OUTPUT_DIR / "explicit_alias_canonicalization_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )


def write_alignment_comparison() -> None:
    if OLD_ALIGNMENT is None or not OLD_ALIGNMENT.exists():
        return
    old = pd.read_csv(OLD_ALIGNMENT, keep_default_na=False)
    new = pd.read_csv(OUTPUT_DIR / "sample_alignment_review.csv", keep_default_na=False)
    keys = ["method", "article", "gt_sample_no"]
    columns = keys + ["status", "gt_sample_name", "pred_sample_no", "pred_sample_name"]
    comparison = old[columns].merge(
        new[columns],
        on=keys,
        how="outer",
        suffixes=("_old", "_final"),
    )
    changed = comparison[
        (comparison["status_old"] != comparison["status_final"])
        | (comparison["pred_sample_no_old"].astype(str) != comparison["pred_sample_no_final"].astype(str))
        | (comparison["pred_sample_name_old"] != comparison["pred_sample_name_final"])
    ].copy()
    changed.to_csv(
        OUTPUT_DIR / "sample_alignment_changes_vs_previous.csv",
        index=False,
        encoding="utf-8-sig",
    )


def write_negative_57_adjudication() -> None:
    if OLD_NEGATIVE_57 is None or not OLD_NEGATIVE_57.exists():
        return
    frame = pd.read_csv(OLD_NEGATIVE_57, keep_default_na=False)
    decisions = []
    notes = []
    for row in frame.to_dict("records"):
        method = row["method"]
        article = str(row["article"])
        gt_no = str(row["gt_sample_no"])
        if method == "Direct LLM":
            decision = "confirmed_correct"
            note = "Source/formulation sequence confirms the pair despite a negative score margin."
        elif method == "Expert Prompt Only" and article == "12":
            decision = "incorrect_pair"
            note = "Prediction contains LEG10 + RP5 + ZB/APP15, not the GT LEG10 + ZB/APP10 formulation."
        elif method == "Expert Prompt Only" and article == "27":
            decision = "incorrect_pair"
            note = "NP28-loading and UV-time experimental series were crossed."
        elif method == "Expert Prompt Only" and article == "335":
            decision = "incorrect_pair"
            note = "Intermediate MH/RP components were paired to final EVA formulations."
        elif method == "Task Decomposition Only" and article == "12":
            decision = "incorrect_pair"
            note = "Prediction contains a conflicting 15% APP/ZB loading and remains prediction-only."
        elif method == "Task Decomposition Only" and article == "422" and gt_no.rstrip(".0") == "9":
            decision = "incorrect_pair"
            note = "The 0 phr compatibilized formulation was swapped with the 90 phr formulation."
        else:
            decision = "confirmed_correct"
            note = "Source sequence, formulation identity, and loading confirm the pair."
        decisions.append(decision)
        notes.append(note)
    frame["manual_adjudication"] = decisions
    frame["manual_note"] = notes
    frame.to_csv(
        OUTPUT_DIR / "negative_margin_nonexact_57_adjudicated.csv",
        index=False,
        encoding="utf-8-sig",
    )


def write_matching_rules(module) -> None:
    scale_rules = [
        {
            "article": article,
            "field": field,
            "accepted_scale_factors": [str(factor) for factor in factors],
        }
        for (article, field), factors in sorted(VERIFIED_SCALE_FACTORS.items())
    ]
    rules = {
        "version": "v18",
        "scored_fields": {
            "entities": module.ENTITY_FIELDS,
            "parts": module.PART_FIELDS,
            "performance": module.PERFORMANCE_FIELDS,
        },
        "object_alignment": {
            "samples": "unordered one-to-one matching within each article using formulation evidence only",
            "entities": "unordered one-to-one matching within an aligned sample",
            "parts": "unordered one-to-one matching within an aligned sample and aligned entity",
            "performance": "one-to-one matching within an aligned sample; unmatched prediction rows are FP",
            "sample_threshold": 0.28,
            "high_confidence_mutual_best_threshold": 0.82,
            "low_score_margin_guard": 0.55,
        },
        "cell_scoring": {
            "equal_nonblank": "TP",
            "unequal_nonblank": "one FP and one FN",
            "GT_only": "FN",
            "prediction_only": "FP",
            "both_blank": "not scored",
        },
        "numeric_tolerance": {
            "LOI": {"absolute": 0.05},
            "pHRR": {"absolute": 0.05},
            "entity_and_part_loading": {"absolute": 0.05},
            "other_numeric_fields": {"absolute": 0.25, "relative": 0.005},
            "multi_value_policy": "same arity and positional comparison; uncertainty uses the central value",
            "verified_scale_rules": scale_rules,
        },
        "loading_equivalence": {
            "explicit_units": ["wt_total", "phr_resin"],
            "priority": "compare explicit-to-explicit values before cross-basis conversion",
            "cross_basis": "an explicit value may match a traceable value derived on the other side",
            "prohibited": "derived-to-derived matches do not receive TP",
            "formulas": {
                "phr_from_wt": "entity_wt_total / matrix_wt_total * 100",
                "wt_from_phr_with_matrix_wt": "entity_phr * matrix_wt_total / 100",
                "wt_from_complete_phr": "entity_phr / (100 + sum(nonmatrix_phr)) * 100",
            },
        },
        "material_equivalence": {
            "accepted": [
                "case/punctuation normalization",
                "documented abbreviations and full names",
                "non-conflicting trade-name/base-material aliases",
                "conservative containment aliases",
                "generic PE entity name plus a matching explicit PE type",
                "a specific PE subtype prediction when GT records only generic PE",
            ],
            "rejected": [
                "conflicting explicit grade/model designators",
                "PE versus LDPE/LLDPE/HDPE/VLDPE",
                "different polyethylene subtypes",
                "containment that adds an identity-changing chemical qualifier",
            ],
        },
        "relation_canonicalization": {
            "aliases": RELATION_ALIASES,
            "policy": "explicit inverse aliases only; no inference from role or modification method",
        },
        "excluded": [
            "sample names and order",
            "entity IDs",
            "entity material type",
            "part loading",
            "SMILES",
            "subjective descriptions",
            "reliability fields",
            "locators and bounding boxes",
            "figure-only l-/w- values",
        ],
    }
    (OUTPUT_DIR / "matching_rules.json").write_text(
        json.dumps(rules, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    global GT_PATH, OUTPUT_DIR, DIRECT_DIR, EXPERT_DIR, SPLIT_DIR, FULL_DIR
    global RERUN_ROOT, OLD_ALIGNMENT, OLD_NEGATIVE_57

    parser = argparse.ArgumentParser(
        description="Evaluate GT50 hierarchical extraction outputs with reproducible matching rules."
    )
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--direct-dir", type=Path, required=True)
    parser.add_argument("--expert-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-rerun-root", type=Path)
    parser.add_argument("--old-alignment", type=Path)
    parser.add_argument("--old-negative-57", type=Path)
    args = parser.parse_args()

    GT_PATH = args.gt.resolve()
    DIRECT_DIR = args.direct_dir.resolve()
    EXPERT_DIR = args.expert_dir.resolve()
    SPLIT_DIR = args.split_dir.resolve()
    FULL_DIR = args.full_dir.resolve()
    OUTPUT_DIR = args.output_dir.resolve()
    RERUN_ROOT = args.split_rerun_root.resolve() if args.split_rerun_root else None
    OLD_ALIGNMENT = args.old_alignment.resolve() if args.old_alignment else None
    OLD_NEGATIVE_57 = args.old_negative_57.resolve() if args.old_negative_57 else None

    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)

    module = load_base_module()
    relation_audit: list[dict] = []
    scope_audit: list[dict] = []
    rejection_rows: list[dict] = []

    module.GT_PATH = GT_PATH
    module.OUT_DIR = OUTPUT_DIR
    module.RELATION_CANON = RELATION_ALIASES.copy()
    module.MOD_METHOD_CANON = {**module.MOD_METHOD_CANON, **METHOD_ALIASES}
    module.ENTITY_FIELDS = [
        field for field in module.ENTITY_FIELDS if field != module.COL["entity_mat_type"]
    ]
    module.PART_FIELDS = [
        field for field in module.PART_FIELDS if field != module.COL["part_amount"]
    ]

    original_field_equal = module.field_equal
    original_entity_pair_score = module.entity_pair_score
    module.numeric_equal = lambda left, right, field: strict_numeric_equal(module, left, right, field)
    module.field_equal = strict_field_equal(module, original_field_equal)
    module.score_pair = strict_score_pair(module)
    module.augment_loadings = lambda frame: augment_loadings_with_provenance(module, frame)
    module.entity_pair_score = patched_entity_pair_score(module, original_entity_pair_score)
    module.part_pair_score = patched_part_pair_score(module)
    module.dedupe_entity_rows = dedupe_entity_rows(module)
    module.dedupe_part_rows = dedupe_part_rows(module)
    module.build_sample_alignment = build_sample_alignment(module, rejection_rows)
    module.score_performance = score_performance(module)
    module.load_methods = lambda: load_methods(module, relation_audit, scope_audit)

    overrides = OUTPUT_DIR / "empty_overrides.json"
    overrides.write_text(
        json.dumps(
            {
                "sample_pairs": [],
                "sample_aliases": [],
                "entity_pairs": [],
                "part_pairs": [],
                "material_equivalences": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    sys.argv = [str(SOURCE_SCORER), "--overrides", str(overrides)]
    module.main()

    pd.DataFrame(rejection_rows).drop_duplicates().to_csv(
        OUTPUT_DIR / "sample_candidate_rejections.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(scope_audit).drop_duplicates().to_csv(
        OUTPUT_DIR / "secondary_condition_scope_exclusions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_relation_audit(relation_audit)
    write_alignment_comparison()
    write_negative_57_adjudication()
    write_matching_rules(module)

    meta_path = OUTPUT_DIR / "evaluation_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(
        {
            "status": "reproducible_primary_evaluation_v18",
            "GT": str(GT_PATH),
            "sample_alignment": "unordered within article; one-to-one constrained maximum-weight matching",
            "sample_conflict_constraints": [
                "matrix presence must agree",
                "recognized matrix material families cannot conflict",
                "explicit specific material designators cannot conflict within the same material family",
                "source-equivalent nitrate and sulfate chemical identities cannot conflict",
                "both sides cannot contain mutually exclusive non-matrix material families",
                "shared non-matrix entity loadings cannot conflict",
                "sample-name loading/irradiation/time/temperature signatures cannot conflict",
                "low-score non-exact pairs require a positive assignment margin",
            ],
            "package_policy": "IFR package and component-resolved representations may align when total non-matrix loading agrees",
            "performance_row_policy": "one-to-one within aligned samples; conflicting extra prediction rows remain FP",
            "numeric_policy": "field-specific tolerances; uncertainty central values; positional equal-length sequences; article-field scale conversions restricted to the verified whitelist in matching_rules.json",
            "loading_policy": "explicit-to-explicit values are checked first; explicit-to-derived cross-basis matches are allowed; derived-to-derived matches are rejected",
            "material_policy": "entity identity combines entity name with explicit PE type; a specific prediction is compatible with generic PE GT, a generic prediction must recover a specific GT subtype through PE type, and containment cannot add identity-changing chemical qualifiers",
            "excluded_primary_fields": [module.COL["entity_mat_type"], module.COL["part_amount"]],
            "relation_policy": "explicit inverse aliases only; no role/method inference",
            "scope_policy": "secondary water-treatment and thermal-aging observations are excluded from the performance scope for every method using the same sample-name rule; exclusions are audited",
            "full_article19_recovery": "sample labels restored from the original saved API JSON; values unchanged",
            "machine_readable_matching_rules": str(OUTPUT_DIR / "matching_rules.json"),
        }
    )
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
