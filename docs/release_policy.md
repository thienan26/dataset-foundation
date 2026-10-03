# Release policy

A release is emitted only after `multicare-data evaluate` reports `PASS`. Empty, provisional or partially reviewed projects are not releases. The audit distinguishes a hard failure from a report-only metric; no aggregate score can hide a failed gate.

The final directory is created atomically and cannot be overwritten. It contains the manifest, development/frozen-test/fold assignments, quarantine and screening audit, source lock, taxonomy, license ledger, per-dimension quality reports, dataset card and SHA-256 checksums. Verify a frozen release with `multicare-data verify-release dataset_v001`.

If the project policy, source release or frozen taxonomy changes, mint `dataset_v002`; preserve all earlier releases and their checksums.

