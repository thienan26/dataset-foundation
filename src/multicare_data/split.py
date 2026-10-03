from __future__ import annotations

import random
from collections import Counter, defaultdict

from .common import PipelineError, Project


def grouped_split(project: Project) -> dict:
    rows = project.rows("data/qc/feasible_candidates.parquet")
    if not rows:
        raise PipelineError("No feasible cases. Complete review and QC first.")
    cfg = project.config("split")
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["split_group"]].append(row)
    class_groups = defaultdict(set)
    for group, members in by_group.items():
        for label in {r["candidate_disease"] for r in members}:
            class_groups[label].add(group)
    folds_n = cfg["development_folds"]
    if any(len(g) < folds_n + 1 for g in class_groups.values()):
        raise PipelineError("Every class needs at least six independent groups for test plus five folds")

    total = Counter(r["candidate_disease"] for r in rows)
    targets = {label: count * cfg["test_fraction"] for label, count in total.items()}
    test_counts = Counter()
    test_groups = set()
    rng = random.Random(cfg["seed"])
    ordered = list(by_group)
    rng.shuffle(ordered)
    ordered.sort(key=lambda g: min(len(class_groups[label]) for label in
                                  {r["candidate_disease"] for r in by_group[g]}))
    for g in ordered:
        labels = Counter(r["candidate_disease"] for r in by_group[g])
        viable = all(len(class_groups[label] - test_groups - {g}) >= folds_n
                     for label in labels)
        if not viable:
            continue
        before = sum((test_counts[l] - targets[l]) ** 2 / max(targets[l], 1) for l in total)
        after = sum((test_counts[l] + labels[l] - targets[l]) ** 2 / max(targets[l], 1) for l in total)
        if after < before:
            test_groups.add(g)
            test_counts.update(labels)
    if not test_groups:
        raise PipelineError("Could not construct a nonempty stratified group test split")
    dev_groups = set(by_group) - test_groups
    dev_counts = Counter(r["candidate_disease"] for g in dev_groups for r in by_group[g])
    if any(test_counts[label] == 0 or dev_counts[label] == 0 for label in total):
        raise PipelineError("A disease class is absent from development or frozen test")

    fold_counts = [Counter() for _ in range(folds_n)]
    fold_members = [[] for _ in range(folds_n)]
    dev_order = list(dev_groups)
    rng.shuffle(dev_order)
    dev_order.sort(key=lambda g: min(len(class_groups[label]) for label in
                                    {r["candidate_disease"] for r in by_group[g]}))
    for g in dev_order:
        group_labels = Counter(r["candidate_disease"] for r in by_group[g])
        score = lambda i, labels=group_labels: sum(
            (fold_counts[i][label] + labels[label]) ** 2 / max(total[label], 1) for label in total)
        target_fold = min(range(folds_n), key=lambda i: (score(i), len(fold_members[i])))
        fold_members[target_fold].append(g)
        fold_counts[target_fold].update(group_labels)
    for label in total:
        if any(count[label] == 0 for count in fold_counts):
            raise PipelineError(f"Class {label} is absent from at least one development fold")

    assignments = []
    for g, members in by_group.items():
        partition = "frozen_test" if g in test_groups else "development"
        fold = None if partition == "frozen_test" else next(i for i, groups in enumerate(fold_members) if g in groups)
        assignments.extend({"case_id": r["case_id"], "article_id": r["article_id"],
                            "split_group": g, "diagnosis_label": r["candidate_disease"],
                            "partition": partition, "fold": fold,
                            "raw_text_sha256": r["raw_text_sha256"],
                            "clinical_text_sha256": r["clinical_text_sha256"],
                            "image_sha256": r["image_sha256"], "pixel_sha256": r["pixel_sha256"]}
                           for r in members)
    project.write("data/splits/assignments.parquet", assignments)
    return {"samples": len(assignments), "groups": len(by_group), "frozen_test_samples": sum(r["partition"] == "frozen_test" for r in assignments),
            "fold_samples": [sum(r["fold"] == i for r in assignments) for i in range(folds_n)]}
