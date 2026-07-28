from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


ARTICLE_INDEX = "article_index"
SAMPLE_INDEX_IN_ARTICLE = "sample_index_in_article"
SAMPLE_NAME = "sample_name"
ARTICLE_TITLE = "article_title"
DOI = "DOI"
SOURCE_ID = "source_id"
SOURCE_FOLDER = "source_folder"
SOURCE_MARKDOWN_PATH = "source_markdown_path"
ARTICLE_UID = "article_uid"
SAMPLE_UID = "sample_uid"
PERFORMANCE_RECORD_INDEX = "performance_record_index"
PERFORMANCE_CONDITION_SIGNATURE = "performance_condition_signature"
PERFORMANCE_RECORD_UID = "performance_record_uid"
FORMULATION_ALIGNMENT_STATUS = "formulation_alignment_status"

SOURCE_COLUMNS = [SOURCE_ID, SOURCE_FOLDER, SOURCE_MARKDOWN_PATH]
IDENTITY_COLUMNS = [ARTICLE_UID, SAMPLE_UID]
PERFORMANCE_IDENTITY_COLUMNS = [
    PERFORMANCE_RECORD_INDEX,
    PERFORMANCE_CONDITION_SIGNATURE,
    PERFORMANCE_RECORD_UID,
    FORMULATION_ALIGNMENT_STATUS,
]

ENTITY_COLUMNS = [
    ARTICLE_INDEX,
    SAMPLE_INDEX_IN_ARTICLE,
    SAMPLE_NAME,
    ARTICLE_TITLE,
    DOI,
    *SOURCE_COLUMNS,
    *IDENTITY_COLUMNS,
    "matrix_system",
    "formulation_measurement_basis",
    "EVA_fraction_in_matrix",
    "PE_fraction_in_matrix",
    "POE_fraction_in_matrix",
    "EVA_VA_content",
    "PE_type",
    "POE_type",
    "matrix",
    "matrix_key_data",
    "melt_flow_index",
    "matrix_grade_information",
    "total_additive_loading",
    "mixing_method",
    "formulation_notes",
    "entity_id",
    "entity_name",
    "entity_role",
    "entity_type",
    "entity_loading",
    "entity_loading_raw",
    "entity_loading_unit",
    "entity_loading_basis",
    "entity_loading_wt_total_formulation",
    "entity_loading_phr_resin",
    "entity_loading_conversion_notes",
    "entity_material_type",
    "entity_grade",
    "entity_profile",
    "entity_figure_locator",
    "entity_bbox",
    "novelty",
    "ai_literature_summary",
]

PART_COLUMNS = [
    ARTICLE_INDEX,
    SAMPLE_INDEX_IN_ARTICLE,
    SAMPLE_NAME,
    ARTICLE_TITLE,
    DOI,
    *SOURCE_COLUMNS,
    *IDENTITY_COLUMNS,
    "entity_id",
    "entity_name",
    "part_name",
    "part_role",
    "part_relation",
    "modification_method",
    "part_ratio",
    "part_loading",
    "part_loading_unit",
    "part_figure_locator",
    "part_bbox",
    "part_smiles",
    "part_smiles_source",
    "part_notes",
]

PERFORMANCE_VALUE_COLUMNS = [
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

PERFORMANCE_COLUMNS = [
    ARTICLE_INDEX,
    SAMPLE_INDEX_IN_ARTICLE,
    SAMPLE_NAME,
    ARTICLE_TITLE,
    DOI,
    *SOURCE_COLUMNS,
    *IDENTITY_COLUMNS,
    *PERFORMANCE_IDENTITY_COLUMNS,
    *PERFORMANCE_VALUE_COLUMNS,
    "performance_notes",
    "novelty",
    "ai_literature_summary",
]

ENTITY_ENRICHMENT_COLUMNS = [
    "entity_smiles",
    "entity_smiles_source",
    "entity_name_original",
    "entity_name_material_category",
    "entity_name_structure_proxy",
    "entity_name_structure_proxy_source",
]

PART_ENTITY_ENRICHMENT_COLUMNS = [
    "entity_smiles",
    "entity_smiles_source",
    "entity_name_original",
    "entity_name_material_category",
    "entity_name_structure_proxy",
    "entity_name_structure_proxy_source",
]

PART_ENRICHMENT_COLUMNS = [
    "part_name_original",
    "part_name_material_category",
    "part_name_structure_proxy",
    "part_name_structure_proxy_source",
]

PATH_COLUMNS = {
    SOURCE_FOLDER,
    SOURCE_MARKDOWN_PATH,
    "original_image_path",
    "collected_image_path",
    "resolved_md_path",
}

RELEASE_ENTITY_COLUMNS = [
    column for column in ENTITY_COLUMNS if column not in PATH_COLUMNS
] + ENTITY_ENRICHMENT_COLUMNS
RELEASE_PART_COLUMNS = [
    column for column in PART_COLUMNS if column not in PATH_COLUMNS
] + PART_ENTITY_ENRICHMENT_COLUMNS + PART_ENRICHMENT_COLUMNS
RELEASE_PERFORMANCE_COLUMNS = [
    column for column in PERFORMANCE_COLUMNS if column not in PATH_COLUMNS
]



def flame_retardant_wide_columns(index: int) -> list[str]:
    prefix = f"flame_retardant_{index}"
    return [
        prefix,
        f"{prefix}_loading_phr",
        f"{prefix}_loading_wt",
        f"{prefix}_type",
        f"{prefix}_grade",
        f"{prefix}_profile",
        f"{prefix}_figure_locator",
        f"{prefix}_bbox",
        f"{prefix}_smiles",
        f"{prefix}_smiles_source",
        f"{prefix}_entity_type",
        f"{prefix}_base_or_core_components",
        f"{prefix}_base_or_core_components_smiles",
        f"{prefix}_base_or_core_components_smiles_source",
        f"{prefix}_shell_grafted_or_modified_components",
        f"{prefix}_shell_grafted_or_modified_components_smiles",
        f"{prefix}_shell_grafted_or_modified_components_smiles_source",
        f"{prefix}_component_ratio",
        f"{prefix}_is_modified",
        f"{prefix}_modified_product_name",
        f"{prefix}_modifier",
        f"{prefix}_modification_method",
        f"{prefix}_modification_ratio",
        f"{prefix}_modifier_loading",
        f"{prefix}_modifier_figure_locator",
        f"{prefix}_modifier_bbox",
        f"{prefix}_modifier_smiles",
        f"{prefix}_modifier_smiles_source",
    ]


MERGED_COLUMNS = [
    "sample_number",
    ARTICLE_INDEX,
    SAMPLE_INDEX_IN_ARTICLE,
    SAMPLE_NAME,
    ARTICLE_TITLE,
    DOI,
    *SOURCE_COLUMNS,
    *IDENTITY_COLUMNS,
    "matrix",
    "matrix_system",
    "formulation_measurement_basis",
    "EVA_fraction_in_matrix",
    "PE_fraction_in_matrix",
    "POE_fraction_in_matrix",
    "EVA_VA_content",
    "PE_type",
    "POE_type",
    "matrix_key_data",
    "melt_flow_index",
    "intrinsic_type_indicator",
    "matrix_grade_information",
    *flame_retardant_wide_columns(1),
    *flame_retardant_wide_columns(2),
    "other_flame_retardant_or_synergist_components",
    "synergistic_flame_retardants",
    "synergistic_flame_retardants_loading_phr",
    "synergistic_flame_retardants_loading_wt",
    "functional_fillers",
    "functional_fillers_loading_phr",
    "functional_fillers_loading_wt",
    "compatibilizers",
    "compatibilizers_loading_phr",
    "compatibilizers_loading_wt",
    "compatibilizers_type",
    "compatibilizers_grade",
    "compatibilizers_figure_locator",
    "compatibilizers_bbox",
    "compatibilizers_smiles",
    "compatibilizers_smiles_source",
    "crosslinking_agents",
    "crosslinking_agents_loading_phr",
    "crosslinking_agents_loading_wt",
    "co_crosslinking_agents",
    "co_crosslinking_agents_loading_phr",
    "co_crosslinking_agents_loading_wt",
    "processing_aids",
    "processing_aids_loading_phr",
    "processing_aids_loading_wt",
    "antioxidants_or_stabilizers",
    "antioxidants_or_stabilizers_loading_phr",
    "antioxidants_or_stabilizers_loading_wt",
    "other_additives",
    "other_additives_loading_phr",
    "other_additives_loading_wt",
    "contains_compatibilizer",
    "is_crosslinked",
    "contains_processing_aid",
    "contains_antioxidant",
    "main_flame_retardant_total_wt",
    "synergistic_flame_retardant_total_wt",
    "functional_filler_total_wt",
    "compatibilizer_total_wt",
    "crosslinking_agent_total_wt",
    "processing_aid_total_wt",
    "antioxidant_or_stabilizer_total_wt",
    "total_additive_loading",
    "mixing_method",
    *PERFORMANCE_VALUE_COLUMNS,
    "formulation_notes",
    "performance_notes",
    "novelty",
    "ai_literature_summary",
]

RELEASE_MERGED_COLUMNS = [column for column in MERGED_COLUMNS if column not in PATH_COLUMNS]


LEGACY_COLUMN_ALIASES: dict[str, str] = {}

def canonical_column_name(column: Any) -> Any:
    if not isinstance(column, str):
        return column
    return LEGACY_COLUMN_ALIASES.get(column, column)


def canonicalize_dataframe_columns(
    dataframe: pd.DataFrame,
    *,
    context: str = "table",
    copy: bool = True,
) -> pd.DataFrame:
    rename: dict[Any, Any] = {}
    seen: dict[Any, Any] = {}
    for column in dataframe.columns:
        canonical = canonical_column_name(column)
        prior = seen.get(canonical)
        if prior is not None and prior != column:
            raise ValueError(
                f"Column collision in {context}: {prior!r} and {column!r} both map to {canonical!r}."
            )
        seen[canonical] = column
        if canonical != column:
            rename[column] = canonical
    result = dataframe.copy() if copy else dataframe
    if rename:
        result.rename(columns=rename, inplace=True)
    return result


def canonicalize_record_keys(record: Mapping[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {}
    for key, value in record.items():
        new_key = canonical_column_name(key)
        if new_key in canonical and key != new_key:
            raise ValueError(f"Record key collision: {key!r} maps to existing key {new_key!r}.")
        canonical[new_key] = value
    return canonical


def legacy_columns_in(columns: Any) -> set[str]:
    return {column for column in columns if column in LEGACY_COLUMN_ALIASES}
