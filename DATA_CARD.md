# Dataset Card

## Overview

This release contains structured records extracted from 535 scientific
articles about EVA and related flame-retardant polymer systems. It links sample
formulations, material entities, internal entity parts, reported performance,
and bibliographic provenance.

The authoritative files are under `data/eva_flame_retardant/`. Counts, column
order, and SHA-256 checksums are recorded in `release_manifest.json`.

## Intended Uses

- formulation-aware information-extraction research;
- record linkage and schema evaluation;
- materials-data curation and quality-control studies;
- literature-to-laboratory research when paired with an independently governed
  laboratory dataset.

The data are not a substitute for safety certification, regulatory testing, or
professional engineering judgment.

## Files

| File | Unit of observation | Rows |
|---|---|---:|
| `formulation_entities.xlsx` | One material entity in one sample | 17,150 |
| `formulation_entity_parts.xlsx` | One internal part of a complex entity | 5,687 |
| `performance_samples_final.xlsx` | One performance record under one reported condition | 5,786 |
| `flame_retardant_data_optimized_final.xlsx` | One denormalized manual-review row | 5,417 |

The three long tables are authoritative. The 147-column wide table is derived
from them for manual inspection and is not a modeling input.

Stable keys are `sample_uid` for sample-level joins,
`performance_record_uid` for performance-record identity, and `article_index`
for article-level grouping. Article-grouped splitting is required to avoid
same-paper leakage.

## Construction

1. Lawfully obtained papers were parsed locally with MinerU.
2. A staged multimodal extraction pipeline recovered formulations, entity-part
   relations, performance values, and source evidence.
3. Text/table performance extraction was supplemented with figure evidence
   where available and then aligned to the formulation roster.
4. Deterministic normalization repaired text encoding, removed exact duplicate
   entity rows, standardized material names, enriched supported SMILES fields,
   and filled missing canonical wt% or phr values only when the formulation
   basis permitted an unambiguous conversion.
5. The three linked long tables were used to rebuild the manual-review wide
   table.

For `part_loading`, the public table retains reported or unambiguously
convertible final-formulation wt% values. Internal-ratio proxy values are not
published as final-formulation loadings; the original `part_ratio` evidence is
preserved.

The repository does not redistribute source PDFs, full-text Markdown, page
images, MinerU intermediates, raw API responses, caches, or run-result JSON.

## Known Limitations

- Measurements across papers are heterogeneous in specimen preparation,
  instrument configuration, test conditions, and reporting conventions.
- Values read from figures may be approximate.
- Complex or blended formulations can exceed the extraction schema's ability
  to recover every relation.
- Minerals, composites, carbon materials, and polymers may not have a single
  chemically meaningful SMILES representation. Structure proxies are labeled
  separately and missing SMILES do not imply missing material identity.
- Hosted model behavior can change, so new API runs are not expected to produce
  byte-identical raw responses.

## Provenance and Privacy

Records retain article titles, DOIs, and evidence fields needed for scientific
audit. Local filesystem paths are removed from the public tables. These are
bibliographic research records rather than a dataset about private individuals.
Users remain responsible for publisher terms and applicable law when obtaining
source documents.

## Licensing and Citation

Authored structured data and annotations are licensed under CC BY 4.0; see
`DATA_LICENSE.md`. This does not grant rights to third-party publications. Use
`CITATION.cff` and cite the exact repository commit used.
