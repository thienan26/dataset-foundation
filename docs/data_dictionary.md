# Data dictionary

`data/raw/multicare/data_dictionary.csv` is the authoritative field reference for MultiCaRe 3.0.1. It is downloaded unchanged, checked against its Zenodo publisher MD5, and retained with the source lock. The normalized tables below add stable names, source SHA-256 values and explicit model-use rules.

| File / field | Source | Type | Nullable | Meaning | Allowed in classifier input? |
|---|---|---|---:|---|---:|
| `articles.parquet.article_id` | MultiCaRe metadata | string | no | PMC article identity and grouping key | no |
| `articles.parquet.title` | MultiCaRe metadata | string | yes | Source article title | no |
| `articles.parquet.license` | MultiCaRe metadata | string | no | Article license as published by source | no |
| `articles.parquet.publication_metadata_json` | MultiCaRe metadata | JSON string | no | All original article metadata, without silently dropping fields | no |
| `cases.parquet.case_id` | MultiCaRe cases | string | no | Patient-case identity; project observation unit | no |
| `cases.parquet.article_id` | MultiCaRe cases | string | no | Parent article; grouping and provenance only | no |
| `cases.parquet.raw_case_text` | MultiCaRe cases | string | no | Immutable case narrative, including diagnosis text | no |
| `cases.parquet.raw_text_sha256` | project | string | no | SHA-256 of exact UTF-8 source narrative | no |
| `cases.parquet.age`, `gender` | MultiCaRe cases | typed source values | yes | De-identified case metadata | no, unless a later protocol explicitly permits it |
| `images.parquet.image_id` | MultiCaRe image table | string | no | Source figure/panel identity | no |
| `images.parquet.case_id` | MultiCaRe image table | string | no | Linked patient case | no |
| `images.parquet.path` | MultiCaRe image file | project-relative path | no | Raw image under `data/raw/multicare/images/` | no |
| `images.parquet.caption` | MultiCaRe image table | string | yes | Figure caption, audit context only | no |
| `images.parquet.license` | MultiCaRe image table | string | no | Image-level rightsholder license | no |
| `labels.parquet.diagnosis_label` | ontology and evidence review | concept ID | no | Single confirmed target class | target only; never input |
| `labels.parquet.diagnosis_evidence` | exact raw narrative span | string | no | Human/AI audit evidence | no |
| `labels.parquet.evidence_start/end` | project | integer | no | Zero-based Python Unicode `[start,end)` offsets into raw case text | no |
| `text_qc.parquet.clinical_text` | deterministic source-prefix cleaning | string | no | Prediagnosis case context, accepted model text | yes, after leakage audit |
| `text_qc.parquet.clinical_text_sha256` | project | string | no | SHA-256 of exact classifier input | no |
| `selected_images.parquet.image_path` | project | project-relative path | no | One selected original image per accepted case | yes, after image audit |
| `selected_images.parquet.image_sha256/pixel_sha256` | project | string | no | Original-byte and normalized-pixel integrity hashes | no |
| `assignments.parquet.partition/fold` | project | enum / integer | no | Group-based frozen-test or development-fold assignment | no |

Missing/unknown source values remain null or carry the explicit source code. Derived values record the stage, configuration/prompt version and hashes in their artifacts; no source field is overwritten.

