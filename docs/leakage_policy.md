# Leakage and split policy

The prediction task observes one prediagnosis case-text prefix and one relevant image. The target, explicit diagnosis evidence, article title, abstract, metadata, image caption, article citation, source filename and postdiagnosis outcome are provenance/annotation-only fields.

Automatic text checks search every discovered in-scope alias and exact diagnosis evidence. Suspected findings stop at human review; an LLM may flag or reject leakage but cannot convert a deterministic failure into a pass. Image review checks actual pixels for relevance and visible diagnostic terminology.

Before splitting, connect samples that share an article, raw/clinical text hash, exact source image, pixel hash or a detected near-image hash. A connected component receives exactly one partition. Compare case IDs, article IDs, group IDs, text hashes, image hashes and pixel hashes across partitions; each intersection must be empty. Leakage gates run again during release verification.

