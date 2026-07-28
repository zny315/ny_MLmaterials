from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.schema import canonicalize_dataframe_columns


ROOT = Path.cwd()
OUT_DIR = ROOT / "evaluation_output"
GT_PATH = ROOT / "GT50.xlsx"

COL = {
    "article": "article_index",
    "sample_no": "sample_index_in_article",
    "sample_name": "sample_name",
    "entity_id": "entity_id",
    "entity_name": "entity_name",
    "entity_role": "entity_role",
    "entity_type": "entity_type",
    "entity_amount": "entity_loading",
    "entity_amount_wt": "entity_loading_wt_total_formulation",
    "entity_amount_phr": "entity_loading_phr_resin",
    "entity_mat_type": "entity_material_type",
    "entity_grade": "entity_grade",
    "part_name": "part_name",
    "part_role": "part_role",
    "part_relation": "part_relation",
    "mod_method": "modification_method",
    "part_ratio": "part_ratio",
    "part_amount": "part_loading",
}

ENTITY_FIELDS = [
    COL["entity_name"],
    COL["entity_role"],
    COL["entity_type"],
    "entity_loading_normalized_wt_or_phr",
    COL["entity_mat_type"],
    COL["entity_grade"],
]

PART_FIELDS = [
    COL["part_name"],
    COL["part_role"],
    COL["part_relation"],
    COL["mod_method"],
    COL["part_ratio"],
    COL["part_amount"],
]

PERFORMANCE_FIELDS = [
    "LOI",
    "UL94_rating",
    "elongation_at_break",
    "tensile_strength",
    "T1",
    "T5",
    "T10",
    "T50",
    "Tmax1",
    "Tmax2",
    "TGA_residue",
    "TTI",
    "cone_heat_flux",
    "cone_specimen_thickness",
    "pHRR",
    "THR",
    "pSPR",
    "TSP",
    "pCOR",
    "pCO2R",
    "cone_residue",
]
VERIFIED_TSP_TSR_ARTICLES = {"38", "44", "273", "438"}

MISSING = {
    "",
    "#",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
    "not reported",
    "not specified",
    "unknown",
}
VISUAL_RE = re.compile(r"^\s*[lw]-", re.IGNORECASE)
NUM_RE = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
PUNCT_RE = re.compile(r"[\s\-_/,;:()\[\]{}]+")
SUBSCRIPT_TRANS = str.maketrans("\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089", "0123456789")

RELATION_CANON = {
    "treated_surface_of": "surface_treated_by",
    "surface_treated_by": "surface_treated_by",
    "intercalated_into": "intercalated_in",
    "intercalated_in": "intercalated_in",
    "doped_into": "doped_in",
    "doped_in": "doped_in",
}
PART_ROLE_CANON = {
    "intercalated_guest": "modifier",
    "complexing_agent": "metal_center",
    "photoinitiator": "other_part",
}
MOD_METHOD_CANON = {
    "microencapsulation": "encapsulation",
    "coordination_complex": "complexation",
    "physical_adsorption": "surface_treatment",
}

MATERIAL_REPLACEMENTS = [
    (r"\bethylen[e]?[-\s]*vinyl[-\s]*acetate(?: copolymer)?\b", "eva"),
    (r"\blinear\s+low\s+density\s+polyethylene\b", "lldpe"),
    (r"\blow\s+density\s+polyethylene\b", "ldpe"),
    (r"\bhigh\s+density\s+polyethylene\b", "hdpe"),
    (r"\bmagnesium\s+hydroxide\b|\bmg\s*\(\s*oh\s*\)\s*2\b|\bmdh\b|\bbrucite\b", "mh"),
    (r"\balumin(?:ium|um)\s+(?:tri)?hydroxide\b|\balumina\s+trihydrate\b|\bath\b", "ath"),
    (r"\bammonium\s+polyphosphate\b", "app"),
    (r"\bexpandable\s+graphite\b|\bexpanded\s+graphite\b", "eg"),
    (r"\bred\s+phosphor(?:us|ous)\b", "rp"),
    (r"\bzinc\s+borate\b", "zb"),
    (
        r"\borgano[-\s]*modified\s+montmorillonite\b|\bmontmorillonite\b|\bommt\b|"
        r"\bcloisite\s*\d+\s*[a-z]?\b|\bnanomer\s*1\.?30t\b|\bnanomer\s*1\.?3t\b",
        "mmt",
    ),
    (r"\bmelamine\s+cyanurate\b", "mca"),
    (r"\bintumescent\s+flame\s+retardant\b", "ifr"),
    (r"\bdicumyl\s+peroxide\b", "dcp"),
    (r"\b3[-\s]*aminopropyltriethoxysilane\b|\bameo\b", "aptes"),
    (r"\btrimethylol\s*propane\s+triacrylate\b", "tmpta"),
    (r"\bpolyethylene\s+grafted\s+with\s+maleic\s+anhydride\b|\bpe[-\s]*g[-\s]*ma(?:h)?\b", "pe-g-mah"),
    (
        r"\bethylene[-\s]*vinyl[-\s]*acetate\s+grafted\s+with\s+maleic\s+anhydride\b|"
        r"\beva[-\s]*g[-\s]*ma(?:h)?\b",
        "eva-g-mah",
    ),
    (r"\battapulgite\b", "palygorskite"),
    (r"\bsilicon\s+dioxide\b|\bsio\s*2\b|\bsilica\s+gel\b", "silica"),
    (r"\bcalcium\s+carbonate\b|\bcaco\s*3\b", "caco3"),
    (r"\bzinc\s+hydroxystannate\b", "zhs"),
    (r"\bdipentaerythritol\b", "dper"),
    (r"\bpentaerythritol\b", "per"),
    (r"\bmelamine\s+phosphate\b", "mp"),
    (r"\bpolypropylene\b", "pp"),
    (r"\bferric\s+pyrophosphate\b", "fepp"),
    (r"\bphytic\s+acid\b", "pa"),
    (r"\bcetyltrimethyl\s+ammonium\s+bromide\b", "ctab"),
    (r"\bhexachlorocyclotriphosphazene\b", "hccp"),
    (r"\btriallyl\s*isocyanurate\b", "taic"),
    (r"\bexpandable\s+graphite\s+eg\d*\b", "eg"),
]

MATERIAL_FAMILY_PATTERNS = [
    ("ifr", r"\bifr\b|\bintumescent\s+flame\s+retardant\b"),
    ("eva", r"\beva(?:\s*\d+[a-z]*)?\b|ethylene[-\s]*vinyl[-\s]*acetate|poly\s*\[?ethylene[-\s]*co[-\s]*\(?vinyl\s*acetate"),
    ("evm", r"\bevm(?:\s*\d+[a-z]*)?\b|ethylene[-\s]*vinyl[-\s]*acetate[^,;]*rubber"),
    ("pe", r"\bpe\b|\bldpe(?:\s*\d+[a-z]*)?\b|\blldpe(?:\s*\d+[a-z]*)?\b|\bhdpe(?:\s*\d+[a-z]*)?\b|\bvldpe\b|polyethylene"),
    ("pp", r"\bpp\b|polypropylene"),
    ("mh", r"\bmh\b|\bmdh\b|magnesium\s+hydroxide|mg\s*\(\s*oh\s*\)\s*2|brucite"),
    ("ath", r"\bath\b|alumin(?:ium|um)\s+(?:tri)?hydroxide|alumina\s+trihydrate"),
    ("app", r"\bapp\b|ammonium\s+polyphosphate"),
    ("eg", r"\beg\d*\b|\bleg\b|\bheg\b|expandable\s+graphite|expanded\s+graphite"),
    ("rp", r"\brp\b|red\s+phosphor(?:us|ous)"),
    ("zb", r"\bzb\b|zinc\s+borate"),
    ("mmt", r"\bmmt\b|\bommt\b|\bomt\b|montmorillonite|organoclay|cloisite\s*\d+\s*[a-z]?|nanomer\s*1\.?3(?:0)?t|na[-\s]*mmt"),
    ("mca", r"\bmca\b|melamine\s+cyanurate"),
    ("silica", r"silica|silicon\s+dioxide|sio\s*2"),
    ("dcp", r"\bdcp\b|dicumyl\s+peroxide"),
    ("per", r"\bper\b|pentaerythritol"),
    ("dper", r"\bdper\b|dipentaerythritol"),
    ("mp", r"\bmp\b|melamine\s+phosphate"),
    ("zhs", r"\bzhs\b|zinc\s+hydroxystannate"),
    ("caco3", r"\bcaco\s*3\b|calcium\s+carbonate"),
    ("pe-g-mah", r"pe[-\s]*g[-\s]*ma(?:h)?|maleic\s+anhydride[^,;]*polyethylene|maleated\s+(?:ll)?d?pe"),
    ("eva-g-mah", r"eva[-\s]*g[-\s]*ma(?:h)?|mah[-\s]*g[-\s]*eva|maleic\s+anhydride[^,;]*ethylene[-\s]*vinyl[-\s]*acetate|maleated\s+eva"),
    ("pgs", r"\bpgs\b|palygorskite|attapulgite"),
    ("ctab", r"\bctab\b|cetyltrimethyl\s+ammonium\s+bromide"),
    ("aptes", r"\baptes\b|\bameo\b|\bkh550\b|aminopropyl\s*triethoxy\s*silane"),
    ("tmpta", r"\btmpta\b|trimethylol\s*propane\s+triacrylate"),
    ("taic", r"\btaic\b|triallyl\s*isocyanurate"),
    ("fepp", r"\bfepp\b|ferric\s+pyrophosphate"),
    ("ldh", r"\bldh\b|layered\s+double\s+hydroxide|hydrotalcite|mg\s*6\s*al\s*2\s*\(oh\)\s*16"),
    ("go", r"\bgo\b|graphene\s+oxide"),
    ("graphite", r"\bgraphite\b"),
    ("hccp", r"\bhccp\b|hexachlorocyclotriphosphazene"),
    ("pzpn", r"\bpzpn\b|polyphosphazene"),
    ("pa", r"\bpa\b|phytic\s+acid"),
    ("cfa", r"\bcfa\b|charring[-\s]*foaming\s+agent"),
    ("pdms", r"\bpdms\b|polydimethylsiloxane"),
    ("black-masterbatch", r"black\s+masterbatch"),
    ("huntite-hydromagnesite", r"huntite\s+hydromagnesite|\bhh\b"),
    ("methyl-tallow-bis-hydroxyethyl-ammonium", r"methyl[-\s]*tallow[-\s]*di\s*\(?2[-\s]*hydroxyethyl\)?\s*ammonium|bis\s*\(?2[-\s]*hydroxyethyl\)?\s*methyl\s*tallow"),
    ("dimethyl-dihydrogenated-tallow-ammonium", r"dimethyl[-\s,]*dihydrogenated\s+tallow[^;]*(?:ammonium|amonia|quaternary)|di[-\s]*methyl\s+di[-\s]*hydro[^;]*tallow[^;]*(?:ammonium|amonia)"),
    ("organic-ammonium-modifier", r"organic\s+(?:quaternary\s+)?ammonium\s+modifier"),
    ("microcapsule-shell", r"microcapsule\s+(?:shell|material)"),
    ("crosslinked-polyurea-shell", r"cross[-\s]*link(?:ed|ing)\s+polyurea(?:\s+shell)?"),
    ("n-octadecyl-trimethyl-amine-chloride", r"n[-\s]*octadecyl\s+trim(?:e|et)hyl\s+amine\s+chloride"),
    ("unknown-surface-treatment", r"(?:unknown|unspecified)\s+surface\s+treatment"),
]


def clean(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value)
    for _ in range(3):
        if not any(marker in text for marker in ["\u00c3", "\u00c2", "\u00e2", "\x82"]):
            break
        try:
            repaired = text.encode("latin1").decode("utf-8")
            if "\ufffd" not in repaired:
                text = repaired
            else:
                break
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
    text = text.translate(SUBSCRIPT_TRANS).replace("\u3000", " ")
    return " ".join(text.strip().split())


def is_missing(value) -> bool:
    return clean(value).lower() in MISSING


def norm_article(value) -> str:
    text = clean(value)
    if text.lower().startswith("article "):
        text = text.split(None, 1)[1]
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def norm_number_text(value) -> str:
    text = clean(value)

    def replace(match):
        try:
            number = Decimal(match.group(0))
        except InvalidOperation:
            return match.group(0)
        out = format(number.normalize(), "f")
        if "." in out:
            out = out.rstrip("0").rstrip(".")
        return "0" if out == "-0" else out

    return NUM_RE.sub(replace, text)


def canon(value) -> str:
    text = norm_number_text(value)
    if text.lower() in MISSING:
        return ""
    text = text.replace("\u00c2\u00b7", "\u00b7").replace("\u00b7", ".").replace("\uff05", "%")
    text = re.sub(r"\bphosphorous\b", "phosphorus", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*%\s*", "%", text)
    return re.sub(r"\s+", " ", text).strip()


def token_key(value) -> str:
    text = clean(value).lower()
    if text in MISSING:
        return ""
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    for pattern, replacement in MATERIAL_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)
    return PUNCT_RE.sub("", text)


def material_equal(left, right) -> tuple[bool, str]:
    left_value = clean(left)
    right_value = clean(right)
    if not left_value and not right_value:
        return True, "both_blank"
    if not left_value or not right_value:
        return False, "one_blank"
    left_key = token_key(left_value)
    right_key = token_key(right_value)
    if left_key == right_key:
        rule = "normalized_exact" if left_value.lower() == right_value.lower() else "documented_alias"
        return True, rule
    left_lower = left_value.lower()
    right_lower = right_value.lower()
    silane_generic = re.compile(r"^(?:silane\s+coupling\s+agent|vinyl\s+silane|methyl\s*silane)$")
    if silane_generic.match(left_lower) and silane_generic.match(right_lower):
        return True, "documented_silane_part_alias"
    left_families = material_families(left_value)
    right_families = material_families(right_value)
    if left_families and left_families == right_families:
        return True, "documented_material_family"
    if len(left_key) >= 4 and len(right_key) >= 4 and (left_key in right_key or right_key in left_key):
        if not left_families or not right_families or left_families == right_families:
            return True, "conservative_containment"
    return False, "different_material"


def material_families(value) -> set[str]:
    text = clean(value).lower().replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    if not text:
        return set()
    if re.search(r"\bifr\b|\bintumescent\s+flame\s+retardant\b", text):
        return {"ifr"}
    families = {family for family, pattern in MATERIAL_FAMILY_PATTERNS if re.search(pattern, text)}
    if "dper" in families:
        families.discard("per")
    if "eg" in families:
        families.discard("graphite")
    if "eva-g-mah" in families:
        families.discard("eva")
    if "pe-g-mah" in families:
        families.discard("pe")
    if "evm" in families:
        families.discard("eva")
    base_families = {
        "eva",
        "evm",
        "pe",
        "pp",
        "mh",
        "ath",
        "app",
        "eg",
        "rp",
        "zb",
        "mmt",
        "mca",
        "silica",
        "caco3",
        "ldh",
        "go",
        "graphite",
        "huntite-hydromagnesite",
    }
    modifier_families = {"ctab", "aptes", "tmpta", "taic", "organic-ammonium-modifier"}
    if families & base_families:
        families -= modifier_families
    return families


def text_similarity(left, right) -> float:
    left_key = token_key(left)
    right_key = token_key(right)
    if not left_key or not right_key:
        return 0.0
    equal, _ = material_equal(left, right)
    if equal:
        return 1.0
    ratio = SequenceMatcher(None, left_key, right_key).ratio()
    left_tokens = set(re.findall(r"[a-z0-9]+", canon(left).lower()))
    right_tokens = set(re.findall(r"[a-z0-9]+", canon(right).lower()))
    jaccard = len(left_tokens & right_tokens) / len(left_tokens | right_tokens) if left_tokens | right_tokens else 0.0
    return max(ratio, jaccard)


def numeric_values(value) -> list[Decimal]:
    values = []
    for raw in NUM_RE.findall(clean(value).replace(",", "")):
        try:
            values.append(Decimal(raw))
        except InvalidOperation:
            continue
    return values


def first_numeric(value):
    values = numeric_values(value)
    return values[0] if values else None


def decimal_text(value: Decimal) -> str:
    out = format(value.normalize(), "f")
    if "." in out:
        out = out.rstrip("0").rstrip(".")
    return "0" if out == "-0" else out


def numeric_close(left: Decimal, right: Decimal, field: str) -> bool:
    diff = abs(left - right)
    if field in {"LOI", "pHRR"}:
        return diff <= Decimal("0.05")
    if field in {COL["entity_amount"], COL["part_amount"], "entity_loading_normalized_wt_or_phr"}:
        return diff <= Decimal("0.05")
    if diff <= Decimal("0.25"):
        return True
    denominator = max(abs(left), abs(right))
    return bool(denominator and diff / denominator <= Decimal("0.005"))


def numeric_equal(left, right, field: str) -> tuple[bool, str]:
    left_values = numeric_values(left)
    right_values = numeric_values(right)
    if not left_values and not right_values:
        return True, "both_blank"
    if not left_values or not right_values:
        return False, "one_blank"
    left_text = clean(left)
    right_text = clean(right)
    if field in PERFORMANCE_FIELDS and re.match(r"^\s*[<>]", left_text):
        bound = left_values[0]
        candidate = right_values[0]
        if (left_text.lstrip().startswith(">") and candidate > bound) or (
            left_text.lstrip().startswith("<") and candidate < bound
        ):
            return True, "GT_reported_bound_satisfied"
    scale_factors = {
        "pSPR": [Decimal("100")],
        "pCOR": [Decimal("1000")],
        "pCO2R": [Decimal("1000")],
        "cone_specimen_thickness": [Decimal("10")],
    }.get(field, [])
    for left_value in left_values:
        for right_value in right_values:
            if numeric_close(left_value, right_value, field):
                return True, "numeric_tolerance"
            for factor in scale_factors:
                if numeric_close(left_value * factor, right_value, field) or numeric_close(
                    left_value, right_value * factor, field
                ):
                    return True, f"documented_scale_x{factor}"
    return False, "numeric_mismatch"


def canonical_category(field: str, value) -> str:
    value = canon(value).lower()
    if field == COL["part_relation"]:
        return RELATION_CANON.get(value, value)
    if field == COL["part_role"]:
        return PART_ROLE_CANON.get(value, value)
    if field == COL["mod_method"]:
        return MOD_METHOD_CANON.get(value, value)
    return value


def field_equal(field: str, left, right) -> tuple[bool, str]:
    left_clean = clean(left)
    right_clean = clean(right)
    if field in {COL["entity_name"], COL["part_name"]}:
        return material_equal(left_clean, right_clean)
    left_value = canon(left_clean)
    right_value = canon(right_clean)
    if not left_value and not right_value:
        return True, "both_blank"
    if not left_value or not right_value:
        return False, "one_blank"
    if field in {
        COL["entity_role"],
        COL["entity_type"],
        COL["entity_mat_type"],
        COL["part_role"],
        COL["part_relation"],
        COL["mod_method"],
        "UL94_rating",
    }:
        equal = canonical_category(field, left_value) == canonical_category(field, right_value)
        return equal, "category_canonical" if equal else "category_mismatch"
    if field == "entity_loading_normalized_wt_or_phr":
        left_parts = parse_loading(left_value)
        right_parts = parse_loading(right_value)
        for key in ("wt", "phr"):
            if key in left_parts and key in right_parts:
                equal, _ = numeric_equal(left_parts[key], right_parts[key], field)
                if equal:
                    return True, f"loading_{key}_equivalent"
        return False, "loading_mismatch"
    if field == COL["part_ratio"]:
        if token_key(left_value) == token_key(right_value):
            return True, "ratio_text_exact"
        left_numbers = numeric_values(left_value)
        right_numbers = numeric_values(right_value)
        if left_numbers and len(left_numbers) == len(right_numbers):
            equal = all(
                numeric_close(left_number, right_number, COL["part_amount"])
                for left_number, right_number in zip(left_numbers, right_numbers)
            )
            return equal, "ratio_numeric_sequence" if equal else "ratio_mismatch"
        return False, "ratio_mismatch"
    if field == COL["part_amount"] or field in PERFORMANCE_FIELDS:
        return numeric_equal(left_value, right_value, field)
    left_key = token_key(left_value)
    right_key = token_key(right_value)
    if left_key == right_key:
        return True, "normalized_exact"
    return False, "text_mismatch"


def sample_uid(row) -> str:
    article = norm_article(row.get(COL["article"], ""))
    sample_no = canon(row.get(COL["sample_no"], "")).lower()
    sample_name = canon(row.get(COL["sample_name"], "")).lower()
    digest = hashlib.sha1(f"{sample_no}|{sample_name}".encode("utf-8")).hexdigest()[:10]
    return f"{article}|{sample_no}|{digest}"


def read_xlsx(path: Path, sheet=0) -> pd.DataFrame:
    return canonicalize_dataframe_columns(
        pd.read_excel(path, sheet_name=sheet, dtype=str, keep_default_na=False).fillna(""),
        context=f"{path}:{sheet}",
    )


def find_full_dir() -> Path:
    candidates = [
        path
        for path in ROOT.iterdir()
        if path.is_dir()
        and path.name.endswith("0606")
        and (path / "formulation_entities.xlsx").exists()
        and (path / "performance_samples_final.xlsx").exists()
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one Full Pipeline directory, found: {candidates}")
    return candidates[0]


def replace_articles(base: pd.DataFrame, rerun_path: Path, article_ids: set[str], sheet=0) -> pd.DataFrame:
    rerun = read_xlsx(rerun_path, sheet)
    base = base.copy()
    base[COL["article"]] = base[COL["article"]].map(norm_article)
    rerun[COL["article"]] = rerun[COL["article"]].map(norm_article)
    return pd.concat(
        [base[~base[COL["article"]].isin(article_ids)], rerun[rerun[COL["article"]].isin(article_ids)]],
        ignore_index=True,
    ).fillna("")


def repair_full_performance_from_raw(frame: pd.DataFrame, full_dir: Path) -> pd.DataFrame:
    article = "19"
    raw_path = full_dir / "api_raw" / "performance_raw_10.1002_app.26028.json"
    if not raw_path.exists():
        return frame
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_samples = raw.get("samples", [])
    indexes = frame.index[frame[COL["article"]].map(norm_article) == article].tolist()
    if len(indexes) != len(raw_samples):
        raise RuntimeError(
            f"Cannot repair Full Article 19 performance labels: {len(indexes)} XLSX rows vs {len(raw_samples)} raw rows"
        )
    repaired = frame.copy()
    for index, raw_sample in zip(indexes, raw_samples):
        repaired.at[index, COL["sample_name"]] = clean(raw_sample.get("sample_name", ""))
    return repaired


def load_methods():
    raise RuntimeError("Use evaluate_gt50.py and provide all input paths through its CLI.")


def matrix_like(row) -> bool:
    role = canon(row.get(COL["entity_role"], "")).lower()
    name = token_key(row.get(COL["entity_name"], ""))
    if role:
        return role == "matrix_polymer"
    return any(token in name for token in ["eva", "ldpe", "lldpe", "hdpe", "poe", "polyethylene", "resin"])


def augment_loadings(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["entity_loading_normalized_wt_or_phr"] = ""
    for _, indexes in frame.groupby("_sample_uid", sort=False).groups.items():
        rows = {index: frame.loc[index].to_dict() for index in indexes}
        matrix_indexes = [index for index in indexes if matrix_like(rows[index])]
        nonmatrix_indexes = [index for index in indexes if index not in matrix_indexes]
        wt_values = {index: first_numeric(rows[index].get(COL["entity_amount_wt"], "")) for index in indexes}
        phr_values = {index: first_numeric(rows[index].get(COL["entity_amount_phr"], "")) for index in indexes}
        matrix_wt = sum((wt_values[index] for index in matrix_indexes if wt_values[index] is not None), Decimal("0"))
        matrix_wt = matrix_wt if matrix_wt > 0 else None
        all_nonmatrix_phr_known = all(phr_values[index] is not None for index in nonmatrix_indexes)
        nonmatrix_phr = [phr_values[index] for index in nonmatrix_indexes if phr_values[index] is not None]
        phr_denominator = Decimal("100") + sum(nonmatrix_phr, Decimal("0")) if all_nonmatrix_phr_known else None
        for index in indexes:
            wt_value = wt_values[index]
            phr_value = phr_values[index]
            if wt_value is None and phr_value is not None:
                if matrix_wt is not None and index not in matrix_indexes:
                    wt_value = phr_value * matrix_wt / Decimal("100")
                elif phr_denominator is not None:
                    wt_value = phr_value / phr_denominator * Decimal("100")
            if phr_value is None and wt_value is not None and matrix_wt is not None:
                phr_value = wt_value / matrix_wt * Decimal("100")
            values = []
            if wt_value is not None:
                values.append(f"wt={decimal_text(wt_value)}")
            if phr_value is not None:
                values.append(f"phr={decimal_text(phr_value)}")
            frame.at[index, "entity_loading_normalized_wt_or_phr"] = ";".join(values)
    return frame


def parse_loading(value) -> dict[str, str]:
    out = {}
    for key in ("wt", "phr"):
        match = re.search(rf"(?:^|[;|]){key}=([^;|]+)", clean(value))
        if match:
            out[key] = match.group(1)
    return out


def loading_similarity(left, right) -> float:
    left_parts = parse_loading(left)
    right_parts = parse_loading(right)
    if not left_parts or not right_parts:
        return 0.0
    for key in ("wt", "phr"):
        if key in left_parts and key in right_parts:
            equal, _ = numeric_equal(left_parts[key], right_parts[key], "entity_loading_normalized_wt_or_phr")
            if equal:
                return 1.0
    return 0.0


def entity_pair_score(left: dict, right: dict) -> float:
    name_score = text_similarity(left.get(COL["entity_name"], ""), right.get(COL["entity_name"], ""))
    identity_equal, _ = material_equal(left.get(COL["entity_name"], ""), right.get(COL["entity_name"], ""))
    if not identity_equal and name_score < 0.70:
        return 0.0
    role_score = float(
        canonical_category(COL["entity_role"], left.get(COL["entity_role"], ""))
        == canonical_category(COL["entity_role"], right.get(COL["entity_role"], ""))
    )
    loading_score = loading_similarity(
        left.get("entity_loading_normalized_wt_or_phr", ""), right.get("entity_loading_normalized_wt_or_phr", "")
    )
    return 0.68 * name_score + 0.14 * role_score + 0.18 * loading_score


def assign_pairs(left_rows: list[dict], right_rows: list[dict], score_function, threshold: float):
    if not left_rows or not right_rows:
        return [], list(range(len(left_rows))), list(range(len(right_rows)))
    source = 0
    left_offset = 1
    right_offset = left_offset + len(left_rows)
    sink = right_offset + len(right_rows)
    graph = [[] for _ in range(sink + 1)]
    edge_references = {}

    def add_edge(start, end, capacity, cost):
        graph[start].append([end, capacity, cost, len(graph[end])])
        graph[end].append([start, 0, -cost, len(graph[start]) - 1])
        return len(graph[start]) - 1

    for left_index in range(len(left_rows)):
        add_edge(source, left_offset + left_index, 1, 0)
    for right_index in range(len(right_rows)):
        add_edge(right_offset + right_index, sink, 1, 0)
    for left_index, left_row in enumerate(left_rows):
        for right_index, right_row in enumerate(right_rows):
            score = float(score_function(left_row, right_row))
            if score < threshold:
                continue
            profit = int(round((score + 1.0) * 1_000_000))
            edge_index = add_edge(left_offset + left_index, right_offset + right_index, 1, -profit)
            edge_references[(left_index, right_index)] = (left_offset + left_index, edge_index, score)

    while True:
        distances = [math.inf] * len(graph)
        parents = [None] * len(graph)
        in_queue = [False] * len(graph)
        distances[source] = 0
        queue = deque([source])
        in_queue[source] = True
        while queue:
            start = queue.popleft()
            in_queue[start] = False
            for edge_index, edge in enumerate(graph[start]):
                end, capacity, cost, _ = edge
                if capacity <= 0:
                    continue
                candidate_distance = distances[start] + cost
                if candidate_distance < distances[end]:
                    distances[end] = candidate_distance
                    parents[end] = (start, edge_index)
                    if not in_queue[end]:
                        queue.append(end)
                        in_queue[end] = True
        if parents[sink] is None or distances[sink] >= 0:
            break
        end = sink
        while end != source:
            start, edge_index = parents[end]
            edge = graph[start][edge_index]
            reverse_index = edge[3]
            edge[1] -= 1
            graph[end][reverse_index][1] += 1
            end = start

    pairs = []
    used_left = set()
    used_right = set()
    for (left_index, right_index), (start, edge_index, score) in edge_references.items():
        if graph[start][edge_index][1] == 0:
            pairs.append((left_index, right_index, score))
            used_left.add(left_index)
            used_right.add(right_index)
    pairs.sort(key=lambda item: (item[0], item[1]))
    unmatched_left = [index for index in range(len(left_rows)) if index not in used_left]
    unmatched_right = [index for index in range(len(right_rows)) if index not in used_right]
    return pairs, unmatched_left, unmatched_right


def entity_rows(frame: pd.DataFrame, sample_id: str) -> list[dict]:
    rows = frame[frame["_sample_uid"] == sample_id].copy()
    return rows.sort_values([COL["entity_role"], COL["entity_name"], COL["entity_id"]]).to_dict("records")


def formulation_summary(rows: list[dict]) -> str:
    values = []
    for row in rows:
        name = clean(row.get(COL["entity_name"], "")) or "?"
        role = clean(row.get(COL["entity_role"], "")) or "?"
        loading = clean(row.get("entity_loading_normalized_wt_or_phr", ""))
        values.append(f"{name}<{role}{',' + loading if loading else ''}>")
    return "; ".join(values)


def greedy_pairs(left_rows: list[dict], right_rows: list[dict], score_function, threshold: float):
    candidates = []
    for left_index, left_row in enumerate(left_rows):
        for right_index, right_row in enumerate(right_rows):
            score = float(score_function(left_row, right_row))
            if score >= threshold:
                candidates.append((score, left_index, right_index))
    candidates.sort(reverse=True)
    pairs = []
    used_left = set()
    used_right = set()
    for score, left_index, right_index in candidates:
        if left_index in used_left or right_index in used_right:
            continue
        pairs.append((left_index, right_index, score))
        used_left.add(left_index)
        used_right.add(right_index)
    return pairs


def sample_pair_score(left_sample: dict, right_sample: dict) -> float:
    left_rows = left_sample["rows"]
    right_rows = right_sample["rows"]
    pairs = greedy_pairs(left_rows, right_rows, entity_pair_score, 0.22)
    if not pairs:
        return 0.0
    composition_score = sum(score for _, _, score in pairs) / max(len(left_rows), len(right_rows))
    count_score = min(len(left_rows), len(right_rows)) / max(len(left_rows), len(right_rows))
    return 0.9 * composition_score + 0.1 * count_score


def sample_records(frame: pd.DataFrame, article: str) -> list[dict]:
    article_frame = frame[frame[COL["article"]] == article]
    records = []
    for sample_id, group in article_frame.groupby("_sample_uid", sort=False):
        first = group.iloc[0]
        records.append(
            {
                "uid": sample_id,
                "sample_no": clean(first.get(COL["sample_no"], "")),
                "sample_name": clean(first.get(COL["sample_name"], "")),
                "rows": group.sort_values([COL["entity_role"], COL["entity_name"], COL["entity_id"]]).to_dict("records"),
            }
        )
    return records


def load_overrides(path: Path):
    if not path.exists():
        return {
            "sample_pairs": [],
            "sample_aliases": [],
            "entity_pairs": [],
            "part_pairs": [],
            "material_equivalences": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def build_sample_alias_map(overrides):
    alias_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for item in overrides.get("sample_aliases", []):
        method = item.get("method", "")
        article = norm_article(item.get("article", ""))
        gt_uid = item.get("gt_uid", "")
        pred_uid = item.get("pred_uid", "")
        if method and article and gt_uid and pred_uid:
            alias_map[method][article][gt_uid].append(pred_uid)
    return alias_map


def override_pairs(overrides, level: str, method: str, article: str):
    return [
        item
        for item in overrides.get(f"{level}_pairs", [])
        if item.get("method") == method and norm_article(item.get("article", "")) == article
    ]


def apply_sample_overrides(candidate_pairs, gt_samples, pred_samples, override_items):
    if not override_items:
        return candidate_pairs
    gt_index = {record["uid"]: index for index, record in enumerate(gt_samples)}
    pred_index = {record["uid"]: index for index, record in enumerate(pred_samples)}
    pairs = []
    for item in override_items:
        gt_uid = item.get("gt_uid", "")
        pred_uid = item.get("pred_uid", "")
        if gt_uid and pred_uid:
            pairs.append((gt_index[gt_uid], pred_index[pred_uid], float(item.get("score", 1.0))))
    return pairs


def build_sample_alignment(selected, gt_entities, methods, overrides):
    records = []
    mappings = {}
    for method, views in methods.items():
        mappings[method] = {}
        pred_entities = views["entities"]
        for article in sorted(selected, key=lambda value: int(value) if value.isdigit() else value):
            gt_samples = sample_records(gt_entities, article)
            pred_samples = sample_records(pred_entities, article)
            score_lookup = {
                (gt_sample["uid"], pred_sample["uid"]): sample_pair_score(gt_sample, pred_sample)
                for gt_sample in gt_samples
                for pred_sample in pred_samples
            }
            gt_labels = defaultdict(list)
            pred_labels = defaultdict(list)
            for index, gt_sample in enumerate(gt_samples):
                gt_labels[token_key(gt_sample["sample_name"])].append(index)
            for index, pred_sample in enumerate(pred_samples):
                pred_labels[token_key(pred_sample["sample_name"])].append(index)
            identifier_pairs = []
            for label, gt_indexes in gt_labels.items():
                pred_indexes = pred_labels.get(label, [])
                if label and len(gt_indexes) == 1 and len(pred_indexes) == 1:
                    gt_index = gt_indexes[0]
                    pred_index = pred_indexes[0]
                    identifier_pairs.append(
                        (
                            gt_index,
                            pred_index,
                            score_lookup[(gt_samples[gt_index]["uid"], pred_samples[pred_index]["uid"])],
                        )
                    )
            identifier_gt = {gt_index for gt_index, _, _ in identifier_pairs}
            identifier_pred = {pred_index for _, pred_index, _ in identifier_pairs}
            remaining_gt_indexes = [index for index in range(len(gt_samples)) if index not in identifier_gt]
            remaining_pred_indexes = [index for index in range(len(pred_samples)) if index not in identifier_pred]
            remaining_gt = [gt_samples[index] for index in remaining_gt_indexes]
            remaining_pred = [pred_samples[index] for index in remaining_pred_indexes]
            score_function = lambda gt_sample, pred_sample: score_lookup[(gt_sample["uid"], pred_sample["uid"])]
            remaining_pairs, _, _ = assign_pairs(remaining_gt, remaining_pred, score_function, 0.28)
            composition_pairs = [
                (remaining_gt_indexes[gt_index], remaining_pred_indexes[pred_index], score)
                for gt_index, pred_index, score in remaining_pairs
            ]
            pairs = identifier_pairs + composition_pairs
            used_gt = {gt_index for gt_index, _, _ in pairs}
            used_pred = {pred_index for _, pred_index, _ in pairs}
            unmatched_gt = [index for index in range(len(gt_samples)) if index not in used_gt]
            unmatched_pred = [index for index in range(len(pred_samples)) if index not in used_pred]
            override_items = override_pairs(overrides, "sample", method, article)
            if override_items:
                pairs = apply_sample_overrides(pairs, gt_samples, pred_samples, override_items)
                used_gt = {left for left, _, _ in pairs}
                used_pred = {right for _, right, _ in pairs}
                unmatched_gt = [index for index in range(len(gt_samples)) if index not in used_gt]
                unmatched_pred = [index for index in range(len(pred_samples)) if index not in used_pred]
            article_mapping = {}
            for gt_index, pred_index, score in pairs:
                gt_sample = gt_samples[gt_index]
                pred_sample = pred_samples[pred_index]
                scores_for_gt = sorted(
                    [score_lookup[(gt_sample["uid"], candidate["uid"])] for candidate in pred_samples], reverse=True
                )
                margin = score - scores_for_gt[1] if len(scores_for_gt) > 1 else score
                exact_label = token_key(gt_sample["sample_name"]) == token_key(pred_sample["sample_name"])
                if score >= 0.82 and margin >= 0.10:
                    review_priority = "low"
                elif score >= 0.62 and margin >= 0.05:
                    review_priority = "medium"
                else:
                    review_priority = "high"
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
                        "gt_formulation": formulation_summary(gt_sample["rows"]),
                        "pred_formulation": formulation_summary(pred_sample["rows"]),
                        "decision_source": (
                            "manual_override"
                            if override_items
                            else ("exact_sample_identifier" if (gt_index, pred_index, score) in identifier_pairs else "composition_candidate")
                        ),
                        "adjudication_note": "",
                    }
                )
            for index in unmatched_gt:
                gt_sample = gt_samples[index]
                records.append(
                    {
                        "method": method,
                        "article": article,
                        "status": "gt_only",
                        "review_priority": "high",
                        "gt_uid": gt_sample["uid"],
                        "pred_uid": "",
                        "gt_sample_no": gt_sample["sample_no"],
                        "pred_sample_no": "",
                        "gt_sample_name": gt_sample["sample_name"],
                        "pred_sample_name": "",
                        "formulation_score": 0,
                        "next_best_margin": 0,
                        "label_exact": False,
                        "gt_formulation": formulation_summary(gt_sample["rows"]),
                        "pred_formulation": "",
                        "decision_source": "composition_candidate",
                        "adjudication_note": "",
                    }
                )
            for index in unmatched_pred:
                pred_sample = pred_samples[index]
                records.append(
                    {
                        "method": method,
                        "article": article,
                        "status": "pred_only",
                        "review_priority": "high",
                        "gt_uid": "",
                        "pred_uid": pred_sample["uid"],
                        "gt_sample_no": "",
                        "pred_sample_no": pred_sample["sample_no"],
                        "gt_sample_name": "",
                        "pred_sample_name": pred_sample["sample_name"],
                        "formulation_score": 0,
                        "next_best_margin": 0,
                        "label_exact": False,
                        "gt_formulation": "",
                        "pred_formulation": formulation_summary(pred_sample["rows"]),
                        "decision_source": "composition_candidate",
                        "adjudication_note": "",
                    }
                )
            mappings[method][article] = article_mapping
    return pd.DataFrame(records), mappings


def entity_uid(method: str, row: dict) -> str:
    raw = "|".join(
        [
            method,
            row.get("_sample_uid", ""),
            clean(row.get(COL["entity_id"], "")),
            clean(row.get(COL["entity_name"], "")),
            clean(row.get(COL["entity_role"], "")),
            clean(row.get("_row_index", "")),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def add_row_ids(frame: pd.DataFrame, method: str, prefix: str) -> pd.DataFrame:
    frame = frame.reset_index(drop=True).copy()
    frame["_row_index"] = frame.index.astype(str)
    if prefix == "entity":
        frame["_entity_uid"] = frame.apply(lambda row: entity_uid(method, row.to_dict()), axis=1)
    else:
        frame[f"_{prefix}_uid"] = [f"{method}:{prefix}:{index}" for index in frame.index]
    return frame


def merge_complementary_rows(target: dict, source: dict, fields: list[str]):
    for field in fields:
        if is_missing(target.get(field, "")) and not is_missing(source.get(field, "")):
            target[field] = source.get(field, "")
    return target


def dedupe_entity_rows(rows: list[dict]):
    merged_rows = []
    for row in rows:
        matched = None
        for candidate in merged_rows:
            same_material = material_equal(
                row.get(COL["entity_name"], ""), candidate.get(COL["entity_name"], "")
            )[0]
            same_role = canonical_category(COL["entity_role"], row.get(COL["entity_role"], "")) == canonical_category(
                COL["entity_role"], candidate.get(COL["entity_role"], "")
            )
            row_loading = row.get("entity_loading_normalized_wt_or_phr", "")
            candidate_loading = candidate.get("entity_loading_normalized_wt_or_phr", "")
            compatible_loading = (
                is_missing(row_loading)
                or is_missing(candidate_loading)
                or loading_similarity(row_loading, candidate_loading) == 1.0
            )
            if same_material and same_role and compatible_loading:
                matched = candidate
                break
        if matched is None:
            copied = dict(row)
            copied["_source_entity_uids"] = [row["_entity_uid"]]
            merged_rows.append(copied)
        else:
            merge_complementary_rows(matched, row, list(row))
            matched["_source_entity_uids"].append(row["_entity_uid"])
    return merged_rows


def build_entity_alignment(gt_entities, methods, sample_mappings, sample_aliases):
    records = []
    mappings = {}
    gt_by_uid = {row["_entity_uid"]: row for row in gt_entities.to_dict("records")}
    pred_by_method = {}
    pred_entity_rep_maps = {}
    for method, views in methods.items():
        mappings[method] = {}
        pred_entity_rep_maps[method] = {}
        pred_entities = views["entities"]
        pred_by_uid = {row["_entity_uid"]: row for row in pred_entities.to_dict("records")}
        pred_by_method[method] = pred_by_uid
        for article, sample_mapping in sample_mappings[method].items():
            for gt_sample, pred_sample in sample_mapping.items():
                gt_rows = gt_entities[gt_entities["_sample_uid"] == gt_sample].to_dict("records")
                pred_sample_ids = [pred_sample] + sample_aliases[method][article].get(gt_sample, [])
                pred_rows = dedupe_entity_rows(
                    pred_entities[pred_entities["_sample_uid"].isin(pred_sample_ids)].to_dict("records")
                )
                for pred_row in pred_rows:
                    pred_by_uid[pred_row["_entity_uid"]] = pred_row
                    for source_uid in pred_row["_source_entity_uids"]:
                        pred_entity_rep_maps[method][source_uid] = pred_row["_entity_uid"]
                pairs, unmatched_gt, unmatched_pred = assign_pairs(gt_rows, pred_rows, entity_pair_score, 0.30)
                for gt_index, pred_index, score in pairs:
                    gt_row = gt_rows[gt_index]
                    pred_row = pred_rows[pred_index]
                    mappings[method][gt_row["_entity_uid"]] = pred_row["_entity_uid"]
                    records.append(
                        {
                            "method": method,
                            "article": article,
                            "gt_sample_uid": gt_sample,
                            "pred_sample_uid": pred_sample,
                            "status": "paired",
                            "gt_entity_uid": gt_row["_entity_uid"],
                            "pred_entity_uid": pred_row["_entity_uid"],
                            "pred_source_entity_uids": ";".join(pred_row["_source_entity_uids"]),
                            "gt_entity_id": gt_row.get(COL["entity_id"], ""),
                            "pred_entity_id": pred_row.get(COL["entity_id"], ""),
                            "gt_entity_name": gt_row.get(COL["entity_name"], ""),
                            "pred_entity_name": pred_row.get(COL["entity_name"], ""),
                            "gt_role": gt_row.get(COL["entity_role"], ""),
                            "pred_role": pred_row.get(COL["entity_role"], ""),
                            "gt_loading": gt_row.get("entity_loading_normalized_wt_or_phr", ""),
                            "pred_loading": pred_row.get("entity_loading_normalized_wt_or_phr", ""),
                            "alignment_score": round(score, 6),
                            "review_priority": "low" if score >= 0.82 else ("medium" if score >= 0.58 else "high"),
                            "adjudication_note": "",
                        }
                    )
                for index in unmatched_gt:
                    gt_row = gt_rows[index]
                    records.append(
                        {
                            "method": method,
                            "article": article,
                            "gt_sample_uid": gt_sample,
                            "pred_sample_uid": pred_sample,
                            "status": "gt_only",
                            "gt_entity_uid": gt_row["_entity_uid"],
                            "pred_entity_uid": "",
                            "pred_source_entity_uids": "",
                            "gt_entity_id": gt_row.get(COL["entity_id"], ""),
                            "pred_entity_id": "",
                            "gt_entity_name": gt_row.get(COL["entity_name"], ""),
                            "pred_entity_name": "",
                            "gt_role": gt_row.get(COL["entity_role"], ""),
                            "pred_role": "",
                            "gt_loading": gt_row.get("entity_loading_normalized_wt_or_phr", ""),
                            "pred_loading": "",
                            "alignment_score": 0,
                            "review_priority": "high",
                            "adjudication_note": "",
                        }
                    )
                for index in unmatched_pred:
                    pred_row = pred_rows[index]
                    records.append(
                        {
                            "method": method,
                            "article": article,
                            "gt_sample_uid": gt_sample,
                            "pred_sample_uid": pred_sample,
                            "status": "pred_only",
                            "gt_entity_uid": "",
                            "pred_entity_uid": pred_row["_entity_uid"],
                            "pred_source_entity_uids": ";".join(pred_row["_source_entity_uids"]),
                            "gt_entity_id": "",
                            "pred_entity_id": pred_row.get(COL["entity_id"], ""),
                            "gt_entity_name": "",
                            "pred_entity_name": pred_row.get(COL["entity_name"], ""),
                            "gt_role": "",
                            "pred_role": pred_row.get(COL["entity_role"], ""),
                            "gt_loading": "",
                            "pred_loading": pred_row.get("entity_loading_normalized_wt_or_phr", ""),
                            "alignment_score": 0,
                            "review_priority": "high",
                            "adjudication_note": "",
                        }
                    )
        recorded_gt = {
            record["gt_entity_uid"]
            for record in records
            if record["method"] == method and record.get("gt_entity_uid")
        }
        recorded_pred = {
            source_uid
            for record in records
            if record["method"] == method and record.get("pred_source_entity_uids")
            for source_uid in record["pred_source_entity_uids"].split(";")
            if source_uid
        }
        for gt_row in gt_entities[~gt_entities["_entity_uid"].isin(recorded_gt)].to_dict("records"):
            records.append(
                {
                    "method": method,
                    "article": norm_article(gt_row.get(COL["article"], "")),
                    "gt_sample_uid": gt_row["_sample_uid"],
                    "pred_sample_uid": "",
                    "status": "gt_only",
                    "gt_entity_uid": gt_row["_entity_uid"],
                    "pred_entity_uid": "",
                    "pred_source_entity_uids": "",
                    "gt_entity_id": gt_row.get(COL["entity_id"], ""),
                    "pred_entity_id": "",
                    "gt_entity_name": gt_row.get(COL["entity_name"], ""),
                    "pred_entity_name": "",
                    "gt_role": gt_row.get(COL["entity_role"], ""),
                    "pred_role": "",
                    "gt_loading": gt_row.get("entity_loading_normalized_wt_or_phr", ""),
                    "pred_loading": "",
                    "alignment_score": 0,
                    "review_priority": "high",
                    "adjudication_note": "parent_sample_unmatched",
                }
            )
        for pred_row in pred_entities[~pred_entities["_entity_uid"].isin(recorded_pred)].to_dict("records"):
            records.append(
                {
                    "method": method,
                    "article": norm_article(pred_row.get(COL["article"], "")),
                    "gt_sample_uid": "",
                    "pred_sample_uid": pred_row["_sample_uid"],
                    "status": "pred_only",
                    "gt_entity_uid": "",
                    "pred_entity_uid": pred_row["_entity_uid"],
                    "pred_source_entity_uids": pred_row["_entity_uid"],
                    "gt_entity_id": "",
                    "pred_entity_id": pred_row.get(COL["entity_id"], ""),
                    "gt_entity_name": "",
                    "pred_entity_name": pred_row.get(COL["entity_name"], ""),
                    "gt_role": "",
                    "pred_role": pred_row.get(COL["entity_role"], ""),
                    "gt_loading": "",
                    "pred_loading": pred_row.get("entity_loading_normalized_wt_or_phr", ""),
                    "alignment_score": 0,
                    "review_priority": "high",
                    "adjudication_note": "parent_sample_unmatched",
                }
            )
    return pd.DataFrame(records), mappings, gt_by_uid, pred_by_method, pred_entity_rep_maps


def link_parts(parts: pd.DataFrame, entities: pd.DataFrame) -> pd.DataFrame:
    parts = parts.copy()
    entities_by_sample = {
        sample: group.to_dict("records") for sample, group in entities.groupby("_sample_uid", sort=False)
    }
    links = []
    reasons = []
    for row in parts.to_dict("records"):
        candidates = entities_by_sample.get(row["_sample_uid"], [])
        entity_id = clean(row.get(COL["entity_id"], ""))
        entity_name = clean(row.get(COL["entity_name"], ""))
        exact_id = [candidate for candidate in candidates if entity_id and clean(candidate.get(COL["entity_id"], "")) == entity_id]
        if len(exact_id) == 1:
            links.append(exact_id[0]["_entity_uid"])
            reasons.append("entity_id")
            continue
        name_matches = [candidate for candidate in candidates if material_equal(entity_name, candidate.get(COL["entity_name"], ""))[0]]
        if len(name_matches) == 1:
            links.append(name_matches[0]["_entity_uid"])
            reasons.append("entity_name")
            continue
        links.append("")
        reasons.append("unlinked")
    parts["_linked_entity_uid"] = links
    parts["_entity_link_reason"] = reasons
    return parts


def part_pair_score(left: dict, right: dict) -> float:
    name_score = text_similarity(left.get(COL["part_name"], ""), right.get(COL["part_name"], ""))
    identity_equal, _ = material_equal(left.get(COL["part_name"], ""), right.get(COL["part_name"], ""))
    if not identity_equal and name_score < 0.70:
        return 0.0
    role_score = float(
        canonical_category(COL["part_role"], left.get(COL["part_role"], ""))
        == canonical_category(COL["part_role"], right.get(COL["part_role"], ""))
    )
    relation_score = float(
        canonical_category(COL["part_relation"], left.get(COL["part_relation"], ""))
        == canonical_category(COL["part_relation"], right.get(COL["part_relation"], ""))
    )
    amount_equal, _ = numeric_equal(left.get(COL["part_amount"], ""), right.get(COL["part_amount"], ""), COL["part_amount"])
    return 0.62 * name_score + 0.14 * role_score + 0.14 * relation_score + 0.10 * float(amount_equal)


def dedupe_part_rows(rows: list[dict]):
    merged_rows = []
    for row in rows:
        matched = None
        for candidate in merged_rows:
            same_material = material_equal(row.get(COL["part_name"], ""), candidate.get(COL["part_name"], ""))[0]
            same_role = canonical_category(COL["part_role"], row.get(COL["part_role"], "")) == canonical_category(
                COL["part_role"], candidate.get(COL["part_role"], "")
            )
            same_relation = canonical_category(
                COL["part_relation"], row.get(COL["part_relation"], "")
            ) == canonical_category(COL["part_relation"], candidate.get(COL["part_relation"], ""))
            row_amount = row.get(COL["part_amount"], "")
            candidate_amount = candidate.get(COL["part_amount"], "")
            compatible_amount = is_missing(row_amount) or is_missing(candidate_amount) or numeric_equal(
                row_amount, candidate_amount, COL["part_amount"]
            )[0]
            if same_material and same_role and same_relation and compatible_amount:
                matched = candidate
                break
        if matched is None:
            copied = dict(row)
            copied["_source_part_uids"] = [row["_part_uid"]]
            merged_rows.append(copied)
        else:
            merge_complementary_rows(matched, row, list(row))
            matched["_source_part_uids"].append(row["_part_uid"])
    return merged_rows


def build_part_alignment(gt_parts, methods, entity_mappings, pred_entity_rep_maps):
    records = []
    mappings = {}
    gt_by_uid = {row["_part_uid"]: row for row in gt_parts.to_dict("records")}
    pred_by_method = {}
    for method, views in methods.items():
        mappings[method] = {}
        pred_parts = views["parts"].copy()
        pred_parts["_linked_entity_uid"] = pred_parts["_linked_entity_uid"].map(
            lambda value: pred_entity_rep_maps[method].get(value, value)
        )
        pred_by_uid = {row["_part_uid"]: row for row in pred_parts.to_dict("records")}
        pred_by_method[method] = pred_by_uid
        gt_entity_to_pred = entity_mappings[method]
        matched_pred_entities = set(gt_entity_to_pred.values())
        for gt_entity, pred_entity in gt_entity_to_pred.items():
            gt_rows = gt_parts[gt_parts["_linked_entity_uid"] == gt_entity].to_dict("records")
            pred_rows = dedupe_part_rows(
                pred_parts[pred_parts["_linked_entity_uid"] == pred_entity].to_dict("records")
            )
            for pred_row in pred_rows:
                pred_by_uid[pred_row["_part_uid"]] = pred_row
            pairs, unmatched_gt, unmatched_pred = assign_pairs(gt_rows, pred_rows, part_pair_score, 0.30)
            for gt_index, pred_index, score in pairs:
                gt_row = gt_rows[gt_index]
                pred_row = pred_rows[pred_index]
                mappings[method][gt_row["_part_uid"]] = pred_row["_part_uid"]
                records.append(
                    {
                        "method": method,
                        "article": norm_article(gt_row.get(COL["article"], "")),
                        "gt_sample_uid": gt_row["_sample_uid"],
                        "pred_sample_uid": pred_row["_sample_uid"],
                        "gt_entity_uid": gt_entity,
                        "pred_entity_uid": pred_entity,
                        "status": "paired",
                        "gt_part_uid": gt_row["_part_uid"],
                        "pred_part_uid": pred_row["_part_uid"],
                        "pred_source_part_uids": ";".join(pred_row["_source_part_uids"]),
                        "gt_part_name": gt_row.get(COL["part_name"], ""),
                        "pred_part_name": pred_row.get(COL["part_name"], ""),
                        "gt_part_role": gt_row.get(COL["part_role"], ""),
                        "pred_part_role": pred_row.get(COL["part_role"], ""),
                        "alignment_score": round(score, 6),
                        "review_priority": "low" if score >= 0.82 else ("medium" if score >= 0.58 else "high"),
                        "adjudication_note": "",
                    }
                )
            for index in unmatched_gt:
                gt_row = gt_rows[index]
                records.append(
                    {
                        "method": method,
                        "article": norm_article(gt_row.get(COL["article"], "")),
                        "gt_sample_uid": gt_row["_sample_uid"],
                        "pred_sample_uid": "",
                        "gt_entity_uid": gt_entity,
                        "pred_entity_uid": pred_entity,
                        "status": "gt_only",
                        "gt_part_uid": gt_row["_part_uid"],
                        "pred_part_uid": "",
                        "pred_source_part_uids": "",
                        "gt_part_name": gt_row.get(COL["part_name"], ""),
                        "pred_part_name": "",
                        "gt_part_role": gt_row.get(COL["part_role"], ""),
                        "pred_part_role": "",
                        "alignment_score": 0,
                        "review_priority": "high",
                        "adjudication_note": "",
                    }
                )
            for index in unmatched_pred:
                pred_row = pred_rows[index]
                records.append(
                    {
                        "method": method,
                        "article": norm_article(pred_row.get(COL["article"], "")),
                        "gt_sample_uid": "",
                        "pred_sample_uid": pred_row["_sample_uid"],
                        "gt_entity_uid": gt_entity,
                        "pred_entity_uid": pred_entity,
                        "status": "pred_only",
                        "gt_part_uid": "",
                        "pred_part_uid": pred_row["_part_uid"],
                        "pred_source_part_uids": ";".join(pred_row["_source_part_uids"]),
                        "gt_part_name": "",
                        "pred_part_name": pred_row.get(COL["part_name"], ""),
                        "gt_part_role": "",
                        "pred_part_role": pred_row.get(COL["part_role"], ""),
                        "alignment_score": 0,
                        "review_priority": "high",
                        "adjudication_note": "",
                    }
                )
        unmatched_gt_entities = set(gt_parts["_linked_entity_uid"]) - set(gt_entity_to_pred)
        unmatched_pred_entities = set(pred_parts["_linked_entity_uid"]) - matched_pred_entities
        for row in gt_parts[gt_parts["_linked_entity_uid"].isin(unmatched_gt_entities)].to_dict("records"):
            records.append(
                {
                    "method": method,
                    "article": norm_article(row.get(COL["article"], "")),
                    "gt_sample_uid": row["_sample_uid"],
                    "pred_sample_uid": "",
                    "gt_entity_uid": row["_linked_entity_uid"],
                    "pred_entity_uid": "",
                    "status": "gt_only",
                    "gt_part_uid": row["_part_uid"],
                    "pred_part_uid": "",
                    "pred_source_part_uids": "",
                    "gt_part_name": row.get(COL["part_name"], ""),
                    "pred_part_name": "",
                    "gt_part_role": row.get(COL["part_role"], ""),
                    "pred_part_role": "",
                    "alignment_score": 0,
                    "review_priority": "high",
                    "adjudication_note": "parent_entity_unmatched",
                }
            )
        for row in pred_parts[pred_parts["_linked_entity_uid"].isin(unmatched_pred_entities)].to_dict("records"):
            records.append(
                {
                    "method": method,
                    "article": norm_article(row.get(COL["article"], "")),
                    "gt_sample_uid": "",
                    "pred_sample_uid": row["_sample_uid"],
                    "gt_entity_uid": "",
                    "pred_entity_uid": row["_linked_entity_uid"],
                    "status": "pred_only",
                    "gt_part_uid": "",
                    "pred_part_uid": row["_part_uid"],
                    "pred_source_part_uids": row["_part_uid"],
                    "gt_part_name": "",
                    "pred_part_name": row.get(COL["part_name"], ""),
                    "gt_part_role": "",
                    "pred_part_role": row.get(COL["part_role"], ""),
                    "alignment_score": 0,
                    "review_priority": "high",
                    "adjudication_note": "parent_entity_unmatched",
                }
            )
    return pd.DataFrame(records).drop_duplicates(["method", "gt_part_uid", "pred_part_uid", "status"]), mappings, gt_by_uid, pred_by_method


def row_nonempty(row: dict, fields: list[str], view: str):
    values = []
    for field in fields:
        value = row.get(field, "")
        if view == "performance" and VISUAL_RE.match(clean(value)):
            continue
        if not is_missing(value):
            values.append(field)
    return values


def score_pair(method, view, article, gt_row, pred_row, fields, context):
    records = []
    for field in fields:
        gt_value = clean(gt_row.get(field, ""))
        pred_value = clean(pred_row.get(field, ""))
        if view == "performance" and VISUAL_RE.match(gt_value):
            continue
        if view == "performance" and VISUAL_RE.match(pred_value):
            pred_value = ""
        gt_present = not is_missing(gt_value)
        pred_present = not is_missing(pred_value)
        if not gt_present and not pred_present:
            continue
        if gt_present and pred_present:
            equal, rule = field_equal(field, gt_value, pred_value)
            if not equal and view == "performance" and field == "TSP" and article in VERIFIED_TSP_TSR_ARTICLES:
                gt_numbers = numeric_values(gt_value)
                pred_numbers = numeric_values(pred_value)
                if any(
                    numeric_close(gt_number * Decimal("100"), pred_number, "TSP")
                    or numeric_close(gt_number, pred_number * Decimal("100"), "TSP")
                    for gt_number in gt_numbers
                    for pred_number in pred_numbers
                ):
                    equal, rule = True, "source_verified_TSR_x_0.01m2_to_TSP"
            if not equal and view == "performance" and field == "TTI":
                gt_fpi = first_numeric(gt_row.get("FPI", "") or gt_row.get("Fire Performance Index", ""))
                pred_fpi = first_numeric(pred_row.get("FPI", "") or pred_row.get("Fire Performance Index", ""))
                gt_phrr = first_numeric(gt_row.get("pHRR", ""))
                pred_phrr = first_numeric(pred_row.get("pHRR", ""))
                derived = []
                if gt_fpi is not None and gt_phrr is not None:
                    derived.append(gt_fpi * gt_phrr)
                if pred_fpi is not None and pred_phrr is not None:
                    derived.append(pred_fpi * pred_phrr)
                gt_tti = first_numeric(gt_value)
                pred_tti = first_numeric(pred_value)
                if (gt_tti is not None and any(numeric_close(gt_tti, value, "TTI") for value in derived)) or (
                    pred_tti is not None and any(numeric_close(pred_tti, value, "TTI") for value in derived)
                ):
                    equal, rule = True, "FPI_x_pHRR"
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


def score_unmatched(method, view, article, row, fields, kind, context):
    records = []
    for field in row_nonempty(row, fields, view):
        records.append(
            {
                "method": method,
                "article": article,
                "view": view,
                **context,
                "field": field,
                "GT": clean(row.get(field, "")) if kind == "gt" else "",
                "PRED": clean(row.get(field, "")) if kind == "pred" else "",
                "outcome": "FN" if kind == "gt" else "FP",
                "equivalence_rule": "unmatched_hierarchical_object",
                "adjudication_note": "",
            }
        )
    return records


def score_entities(entity_alignment, gt_by_uid, pred_by_method):
    records = []
    for row in entity_alignment.to_dict("records"):
        method = row["method"]
        article = row["article"]
        context = {
            "gt_sample_uid": row.get("gt_sample_uid", ""),
            "pred_sample_uid": row.get("pred_sample_uid", ""),
            "gt_object_uid": row.get("gt_entity_uid", ""),
            "pred_object_uid": row.get("pred_entity_uid", ""),
        }
        if row["status"] == "paired":
            records.extend(
                score_pair(
                    method,
                    "entities",
                    article,
                    gt_by_uid[row["gt_entity_uid"]],
                    pred_by_method[method][row["pred_entity_uid"]],
                    ENTITY_FIELDS,
                    context,
                )
            )
        elif row["status"] == "gt_only":
            records.extend(
                score_unmatched(
                    method,
                    "entities",
                    article,
                    gt_by_uid[row["gt_entity_uid"]],
                    ENTITY_FIELDS,
                    "gt",
                    context,
                )
            )
        else:
            records.extend(
                score_unmatched(
                    method,
                    "entities",
                    article,
                    pred_by_method[method][row["pred_entity_uid"]],
                    ENTITY_FIELDS,
                    "pred",
                    context,
                )
            )
    return records


def score_parts(part_alignment, gt_by_uid, pred_by_method):
    records = []
    for row in part_alignment.to_dict("records"):
        method = row["method"]
        article = row["article"]
        context = {
            "gt_sample_uid": row.get("gt_sample_uid", ""),
            "pred_sample_uid": row.get("pred_sample_uid", ""),
            "gt_object_uid": row.get("gt_part_uid", ""),
            "pred_object_uid": row.get("pred_part_uid", ""),
        }
        if row["status"] == "paired":
            records.extend(
                score_pair(
                    method,
                    "parts",
                    article,
                    gt_by_uid[row["gt_part_uid"]],
                    pred_by_method[method][row["pred_part_uid"]],
                    PART_FIELDS,
                    context,
                )
            )
        elif row["status"] == "gt_only":
            records.extend(
                score_unmatched(
                    method,
                    "parts",
                    article,
                    gt_by_uid[row["gt_part_uid"]],
                    PART_FIELDS,
                    "gt",
                    context,
                )
            )
        else:
            records.extend(
                score_unmatched(
                    method,
                    "parts",
                    article,
                    pred_by_method[method][row["pred_part_uid"]],
                    PART_FIELDS,
                    "pred",
                    context,
                )
            )
    return records


def link_performance(performance: pd.DataFrame, entities: pd.DataFrame) -> pd.DataFrame:
    performance = performance.copy()
    entity_samples = {}
    for article, group in entities.groupby(COL["article"], sort=False):
        sample_rows = group.drop_duplicates("_sample_uid")
        entity_samples[article] = sample_rows.to_dict("records")
    links = []
    reasons = []
    for row in performance.to_dict("records"):
        candidates = entity_samples.get(row[COL["article"]], [])
        exact_uid = [candidate for candidate in candidates if candidate["_sample_uid"] == row["_sample_uid"]]
        if len(exact_uid) == 1:
            links.append(exact_uid[0]["_sample_uid"])
            reasons.append("exact_sample_identifier")
            continue
        name = token_key(row.get(COL["sample_name"], ""))
        name_matches = [candidate for candidate in candidates if name and token_key(candidate.get(COL["sample_name"], "")) == name]
        if len(name_matches) == 1:
            links.append(name_matches[0]["_sample_uid"])
            reasons.append("unique_sample_label")
            continue
        links.append("")
        reasons.append("unlinked")
    performance["_linked_sample_uid"] = links
    performance["_sample_link_reason"] = reasons
    return performance


def performance_row_score(left: dict, right: dict) -> float:
    left_name = token_key(left.get(COL["sample_name"], ""))
    right_name = token_key(right.get(COL["sample_name"], ""))
    if left_name and right_name and left_name == right_name:
        return 1.0
    return 0.5


def performance_row_in_scope(row: dict) -> bool:
    article = norm_article(row.get(COL["article"], ""))
    name = clean(row.get(COL["sample_name"], "")).lower()
    notes = clean(row.get("performance_notes", "")).lower()
    if article == "236" and (
        "water treatment" in name
        or "conditioned water-resistance" in notes
        or "not in formulation roster" in notes
    ):
        return False
    return True


def merge_performance_rows(rows: list[dict]):
    if not rows:
        return []
    merged = dict(rows[0])
    merged["_source_performance_uids"] = [rows[0]["_performance_uid"]]
    for row in rows[1:]:
        for field in PERFORMANCE_FIELDS:
            current = clean(merged.get(field, ""))
            candidate = clean(row.get(field, ""))
            if (is_missing(current) or VISUAL_RE.match(current)) and not is_missing(candidate) and not VISUAL_RE.match(candidate):
                merged[field] = candidate
        merged["_source_performance_uids"].append(row["_performance_uid"])
    return [merged]


def score_performance(gt_perf, methods, sample_mappings, sample_aliases):
    records = []
    for method, views in methods.items():
        pred_perf = views["performance"]
        pred_perf = pred_perf[pred_perf.apply(lambda row: performance_row_in_scope(row.to_dict()), axis=1)].copy()
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
                if len(gt_rows) == 1 and len(pred_sample_ids) > 1:
                    pred_rows = merge_performance_rows(pred_rows)
                else:
                    for pred_row in pred_rows:
                        pred_row["_source_performance_uids"] = [pred_row["_performance_uid"]]
                pairs, unmatched_gt, unmatched_pred = assign_pairs(gt_rows, pred_rows, performance_row_score, 0.49)
                for gt_index, pred_index, _ in pairs:
                    gt_row = gt_rows[gt_index]
                    pred_row = pred_rows[pred_index]
                    used_gt.add(gt_row["_performance_uid"])
                    used_pred.update(pred_row["_source_performance_uids"])
                    context = {
                        "gt_sample_uid": gt_sample,
                        "pred_sample_uid": pred_sample,
                        "gt_object_uid": gt_row["_performance_uid"],
                        "pred_object_uid": pred_row["_performance_uid"],
                    }
                    records.extend(
                        score_pair(method, "performance", article, gt_row, pred_row, PERFORMANCE_FIELDS, context)
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
                        score_unmatched(method, "performance", article, gt_row, PERFORMANCE_FIELDS, "gt", context)
                    )
                for index in unmatched_pred:
                    pred_row = pred_rows[index]
                    used_pred.update(pred_row["_source_performance_uids"])
                    context = {
                        "gt_sample_uid": gt_sample,
                        "pred_sample_uid": pred_sample,
                        "gt_object_uid": "",
                        "pred_object_uid": pred_row["_performance_uid"],
                    }
                    records.extend(
                        score_unmatched(method, "performance", article, pred_row, PERFORMANCE_FIELDS, "pred", context)
                    )
        for gt_row in gt_perf[~gt_perf["_performance_uid"].isin(used_gt)].to_dict("records"):
            context = {
                "gt_sample_uid": gt_row.get("_linked_sample_uid", ""),
                "pred_sample_uid": "",
                "gt_object_uid": gt_row["_performance_uid"],
                "pred_object_uid": "",
            }
            records.extend(
                score_unmatched(
                    method,
                    "performance",
                    norm_article(gt_row.get(COL["article"], "")),
                    gt_row,
                    PERFORMANCE_FIELDS,
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
                score_unmatched(
                    method,
                    "performance",
                    norm_article(pred_row.get(COL["article"], "")),
                    pred_row,
                    PERFORMANCE_FIELDS,
                    "pred",
                    context,
                )
            )
    return records


def metric_counts(frame: pd.DataFrame):
    tp = int((frame["outcome"] == "TP").sum())
    fp = int(frame["outcome"].isin(["FP", "FP+FN"]).sum())
    fn = int(frame["outcome"].isin(["FN", "FP+FN"]).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return tp, fp, fn, precision, recall, f1


def summarize_scores(cell_scores: pd.DataFrame):
    rows = []
    for method in cell_scores["method"].drop_duplicates():
        method_frame = cell_scores[cell_scores["method"] == method]
        for view in ["entities", "parts", "performance", "all"]:
            view_frame = method_frame if view == "all" else method_frame[method_frame["view"] == view]
            tp, fp, fn, precision, recall, f1 = metric_counts(view_frame)
            rows.append(
                {
                    "method": method,
                    "view": view,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "precision": precision,
                    "recall": recall,
                    "F1": f1,
                }
            )
    return pd.DataFrame(rows)


def summarize_fields(cell_scores: pd.DataFrame):
    rows = []
    for (method, view, field), group in cell_scores.groupby(["method", "view", "field"], sort=False):
        tp, fp, fn, precision, recall, f1 = metric_counts(group)
        rows.append(
            {
                "method": method,
                "view": view,
                "field": field,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": precision,
                "recall": recall,
                "F1": f1,
            }
        )
    return pd.DataFrame(rows)


def summarize_articles(cell_scores: pd.DataFrame):
    rows = []
    for (method, article, view), group in cell_scores.groupby(["method", "article", "view"], sort=False):
        tp, fp, fn, precision, recall, f1 = metric_counts(group)
        rows.append(
            {
                "method": method,
                "article": article,
                "view": view,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": precision,
                "recall": recall,
                "F1": f1,
            }
        )
    article_view = pd.DataFrame(rows)
    all_rows = []
    for (method, article), group in cell_scores.groupby(["method", "article"], sort=False):
        tp, fp, fn, precision, recall, f1 = metric_counts(group)
        all_rows.append(
            {
                "method": method,
                "article": article,
                "view": "all",
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": precision,
                "recall": recall,
                "F1": f1,
            }
        )
    return pd.concat([article_view, pd.DataFrame(all_rows)], ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overrides", type=Path, default=OUT_DIR / "manual_overrides.json")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected, gt, methods = load_methods()
    gt["entities"] = add_row_ids(augment_loadings(gt["entities"]), "GT", "entity")
    gt["parts"] = add_row_ids(gt["parts"], "GT", "part")
    gt["performance"] = add_row_ids(gt["performance"], "GT", "performance")
    for method, views in methods.items():
        views["entities"] = add_row_ids(augment_loadings(views["entities"]), method, "entity")
        views["parts"] = add_row_ids(views["parts"], method, "part")
        views["performance"] = add_row_ids(views["performance"], method, "performance")
    gt["parts"] = link_parts(gt["parts"], gt["entities"])
    gt["performance"] = link_performance(gt["performance"], gt["entities"])
    for views in methods.values():
        views["parts"] = link_parts(views["parts"], views["entities"])
        views["performance"] = link_performance(views["performance"], views["entities"])
    overrides = load_overrides(args.overrides)
    sample_aliases = build_sample_alias_map(overrides)
    sample_alignment, sample_mappings = build_sample_alignment(
        selected, gt["entities"], methods, overrides
    )
    entity_alignment, entity_mappings, gt_entities_by_uid, pred_entities_by_method, pred_entity_rep_maps = build_entity_alignment(
        gt["entities"], methods, sample_mappings, sample_aliases
    )
    part_alignment, part_mappings, gt_parts_by_uid, pred_parts_by_method = build_part_alignment(
        gt["parts"], methods, entity_mappings, pred_entity_rep_maps
    )
    cell_records = score_entities(entity_alignment, gt_entities_by_uid, pred_entities_by_method)
    cell_records.extend(score_parts(part_alignment, gt_parts_by_uid, pred_parts_by_method))
    cell_records.extend(score_performance(gt["performance"], methods, sample_mappings, sample_aliases))
    cell_scores = pd.DataFrame(cell_records)
    summary = summarize_scores(cell_scores)
    field_summary = summarize_fields(cell_scores)
    article_summary = summarize_articles(cell_scores)
    sample_alignment.to_csv(OUT_DIR / "sample_alignment_review.csv", index=False, encoding="utf-8-sig")
    entity_alignment.to_csv(OUT_DIR / "entity_alignment_review.csv", index=False, encoding="utf-8-sig")
    part_alignment.to_csv(OUT_DIR / "part_alignment_review.csv", index=False, encoding="utf-8-sig")
    cell_scores.to_csv(OUT_DIR / "cell_level_scores.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "metric_summary_candidate.csv", index=False, encoding="utf-8-sig")
    field_summary.to_csv(OUT_DIR / "field_metrics_candidate.csv", index=False, encoding="utf-8-sig")
    article_summary.to_csv(OUT_DIR / "article_metrics_candidate.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(overrides.get("sample_aliases", [])).to_csv(
        OUT_DIR / "duplicate_sample_alias_review.csv", index=False, encoding="utf-8-sig"
    )
    review_queue = pd.concat(
        [
            sample_alignment[sample_alignment["review_priority"].isin(["high", "medium"])].assign(level="sample"),
            entity_alignment[entity_alignment["review_priority"].isin(["high", "medium"])].assign(level="entity"),
            part_alignment[part_alignment["review_priority"].isin(["high", "medium"])].assign(level="part"),
        ],
        ignore_index=True,
        sort=False,
    )
    review_queue.to_csv(OUT_DIR / "manual_review_queue.csv", index=False, encoding="utf-8-sig")
    meta = {
        "GT": str(GT_PATH),
        "selected_articles": len(selected),
        "methods": list(methods),
        "fields": {
            "entities": ENTITY_FIELDS,
            "parts": PART_FIELDS,
            "performance": PERFORMANCE_FIELDS,
        },
        "principles": [
            "Samples are unordered within each article.",
            "Sample candidates use formulation entities, roles, and loadings; performance values are never anchors.",
            "Entities are aligned only within an aligned sample.",
            "Parts are aligned only within an aligned sample and aligned entity.",
            "Performance is scored only within an aligned sample.",
            "Explicit source-table duplicate aliases are merged only after manual composition review.",
            "Sample names, sequence numbers, and entity IDs are not scored fields.",
            "SMILES, subjective descriptions, reliability, locators/BBox, and figure-only l-/w- values are excluded.",
        ],
        "status": "candidate_until_manual_review_complete",
    }
    (OUT_DIR / "evaluation_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.overrides.exists():
        args.overrides.write_text(
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
    print(summary.to_string(index=False))
    print("\nReview queue:", len(review_queue))
    print("Output:", OUT_DIR)


if __name__ == "__main__":
    main()
