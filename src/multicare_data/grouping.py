from __future__ import annotations

from collections import defaultdict

from .common import PipelineError, Project, digest
from .sample import single_targets


class UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}
        self.rank = dict.fromkeys(values, 0)

    def find(self, value):
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            value, self.parent[value] = self.parent[value], root
        return root

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def group(project: Project) -> dict:
    samples = single_targets(project)
    ids = [r["case_id"] for r in samples]
    uf = UnionFind(ids)
    image_by_case = {r["case_id"]: r for r in project.rows("data/qc/selected_images.parquet")}
    for field in ("article_id", "raw_text_sha256", "clinical_text_sha256"):
        index = defaultdict(list)
        for r in samples:
            value = r.get(field)
            if value:
                index[value].append(r["case_id"])
        for values in index.values():
            for other in values[1:]:
                uf.union(values[0], other)
    image_indexes = defaultdict(list)
    perceptual_buckets = defaultdict(set)
    for cid, row in image_by_case.items():
        if cid in uf.parent:
            for field in ("image_sha256", "pixel_sha256"):
                if row.get(field):
                    image_indexes[field, row[field]].append(cid)
            phash = row.get("perceptual_hash")
            if phash:
                try:
                    bits = int(phash, 16)
                    # With four fixed 16-bit bands, any 64-bit hashes at Hamming <= 4
                    # must share at least one band. Exact check below controls false positives.
                    for band in range(4):
                        perceptual_buckets[band, (bits >> (band * 16)) & 0xFFFF].add(cid)
                except ValueError:
                    raise PipelineError(f"Invalid perceptual hash for {cid}")
    for values in image_indexes.values():
        for other in values[1:]:
            uf.union(values[0], other)
    for (band, _), members in perceptual_buckets.items():
        members = sorted(members)
        for i, left in enumerate(members):
            for right in members[i + 1:]:
                a, b = int(image_by_case[left]["perceptual_hash"], 16), int(image_by_case[right]["perceptual_hash"], 16)
                if (a ^ b).bit_count() <= project.config("qc")["near_image_hamming"]:
                    uf.union(left, right)
    components = defaultdict(list)
    for cid in ids:
        components[uf.find(cid)].append(cid)
    normalized = {min(members): sorted(members) for members in components.values()}
    label_by_case = {r["case_id"]: r["candidate_disease"] for r in samples}
    conflicts, rows = [], []
    for members in normalized.values():
        labels = sorted({label_by_case[cid] for cid in members})
        conflict = len(labels) > 1
        if conflict:
            conflicts.append({"split_group": digest("\n".join(members)), "case_ids": members,
                              "disease_labels": labels, "reason": "DUPLICATE_LABEL_CONFLICT"})
        for cid in members:
            rows.append({"case_id": cid, "split_group": digest("\n".join(members)),
                         "component_size": len(members), "disease_labels": labels,
                         "duplicate_conflict": conflict})
    project.write("data/groups/groups.parquet", rows)
    project.write("data/qc/duplicate_conflicts.parquet", conflicts)
    return {"cases": len(rows), "independent_groups": len(normalized), "duplicate_conflicts": len(conflicts)}


def feasibility(project: Project) -> dict:
    samples = {r["case_id"]: r for r in single_targets(project)}
    texts = {r["case_id"]: r for r in project.rows("data/qc/text_qc.parquet")}
    images = {r["case_id"]: r for r in project.rows("data/qc/selected_images.parquet")}
    groups = {r["case_id"]: r for r in project.rows("data/groups/groups.parquet")}
    humans = {r["case_id"]: r for r in project.rows("data/reviews/human_reviews.parquet", optional=True)}
    resolved = {r["case_id"]: r for r in project.rows("data/reviews/adjudication.parquet")}
    eligible = []
    reasons = defaultdict(int)
    for cid, row in samples.items():
        reason = None
        text, image, component = texts.get(cid), images.get(cid), groups.get(cid)
        review = resolved.get(cid, {})
        if row["source_license"] in {"UNKNOWN", "NO-CC CODE", "", None}:
            reason = "LICENSE_UNKNOWN"
        elif image and image.get("license") in {"UNKNOWN", "NO-CC CODE", "", None}:
            reason = "IMAGE_LICENSE_UNKNOWN"
        elif not text or not text.get("text_usable"):
            reason = "TEXT_QC_PENDING_OR_FAILED"
        elif not image or not image.get("image_usable"):
            reason = "IMAGE_QC_PENDING_OR_FAILED"
        elif not review.get("verified"):
            reason = "LABEL_REVIEW_PENDING_OR_REJECTED"
        elif component.get("duplicate_conflict"):
            reason = "DUPLICATE_CONFLICT"
        elif humans.get(cid) and (humans[cid].get("human_decision") == "reject"
                                  or not humans[cid].get("human_evidence_valid")
                                  or humans[cid].get("text_leakage_found")
                                  or humans[cid].get("image_label_leakage")
                                  or humans[cid].get("image_relevance") not in {"diagnostic", "supportive"}):
            reason = "HUMAN_AUDIT_FAILED"
        if reason:
            reasons[reason] += 1
        else:
            eligible.append({**row, **text, **image, "split_group": component["split_group"],
                             "review_status": review["review_status"],
                             "human_decision": humans.get(cid, {}).get("human_decision", "not_sampled")})
    by_class = defaultdict(list)
    for row in eligible:
        by_class[row["candidate_disease"]].append(row)
    cfg = project.config("split")
    classes, report = {}, []
    for concept_id, rows in sorted(by_class.items()):
        independent = len({r["split_group"] for r in rows})
        accepted = independent >= cfg["minimum_groups_per_class"] and len(rows) >= cfg["minimum_cases_per_class"]
        classes[concept_id] = {"disease": rows[0]["disease"], "concept_id": concept_id,
                               "usable_cases": len(rows), "independent_groups": independent,
                               "image_supported_cases": len(rows), "status": "in_v001" if accepted else "insufficient_support"}
        report.append(classes[concept_id])
    project.write("reports/disease_feasibility.parquet", report)
    kept = {cid for cid, value in classes.items() if value["status"] == "in_v001"}
    output = [r for r in eligible if r["candidate_disease"] in kept]
    project.write("data/qc/feasible_candidates.parquet", output)
    return {"candidate_cases": len(samples), "eligible_cases": len(eligible), "feasible_classes": len(kept),
            "classes": {v["disease"]: {"cases": v["usable_cases"], "groups": v["independent_groups"],
                                       "status": v["status"]} for v in classes.values()},
            "pending_or_excluded": dict(reasons)}
