from __future__ import annotations

import re
from collections import Counter, defaultdict

import ahocorasick

from .acquire import download
from .common import PipelineError, Project, file_hash, read_json, write_json


def acquire_ontology(project: Project) -> dict:
    cfg = project.config("scope")
    url = f"https://raw.githubusercontent.com/{cfg['ontology_repository']}/{cfg['ontology_commit']}/{cfg['ontology_path']}"
    path = project.path("taxonomy/candidates/doid.obo")
    lock_path = project.path("taxonomy/candidates/ontology_lock.json")
    if lock_path.exists():
        lock = read_json(lock_path)
        if lock["commit"] != cfg["ontology_commit"] or not path.exists() or file_hash(path) != lock["sha256"]:
            raise PipelineError("Ontology changed after pinning")
        return lock
    entry = download(url, path)
    entry.update({"commit": cfg["ontology_commit"], "license": "CC0-1.0"})
    write_json(lock_path, entry, immutable=True)
    return entry


def parse_obo(text: str, cfg: dict) -> list[dict]:
    terms = {}
    for block in re.split(r"(?m)^\[Term\]\s*$", text)[1:]:
        block = block.split("[Typedef]")[0]
        if re.search(r"(?m)^is_obsolete: true", block):
            continue
        identity = re.search(r"(?m)^id: (DOID:\d+)", block)
        name = re.search(r"(?m)^name: (.+)$", block)
        if not identity or not name:
            continue
        cid, label = identity[1], name[1].strip()
        aliases = [label] + re.findall(r'(?m)^synonym: "([^"\n]+)" EXACT', block)
        aliases += cfg.get("extra_aliases", {}).get(cid, [])
        terms[cid] = {"concept_id": cid, "disease": label,
                      "aliases": sorted({a for a in aliases if len(a) >= cfg["minimum_alias_length"]}),
                      "parents": re.findall(r"(?m)^is_a: (DOID:\d+)", block)}
    memo = {}

    def ancestors(cid, visiting=frozenset()):
        if cid in memo:
            return memo[cid]
        if cid in visiting:
            raise PipelineError("Cycle in ontology")
        parents = terms.get(cid, {}).get("parents", [])
        result = set(parents)
        for parent in parents:
            result.update(ancestors(parent, visiting | {cid}))
        memo[cid] = result
        return result

    roots = set(cfg["infectious_roots"])
    if not roots.issubset(terms):
        raise PipelineError("Infectious scope root missing from ontology")
    for cid, row in terms.items():
        row["ancestors"] = sorted(ancestors(cid))
        row["in_scope"] = bool((set(row["ancestors"]) | {cid}) & roots) and cid not in cfg["excluded_concepts"]
        row["tropical"] = True if cid in cfg.get("tropical_concepts", []) else None
        row["tropical_status"] = "curated_yes" if row["tropical"] else "unknown"
        row["scope_reason"] = "INFECTIOUS_DESCENDANT" if row["in_scope"] else "OUT_OF_SCOPE"
    return sorted(terms.values(), key=lambda r: r["concept_id"])


class Matcher:
    def __init__(self, concepts: list[dict]):
        aliases = defaultdict(set)
        for term in concepts:
            for alias in term["aliases"]:
                aliases[alias.lower()].add(term["concept_id"])
        self.automaton = ahocorasick.Automaton()
        for alias, ids in aliases.items():
            self.automaton.add_word(alias, (alias, sorted(ids)))
        self.active = bool(aliases)
        if self.active:
            self.automaton.make_automaton()

    def find(self, text: str) -> list[dict]:
        if not self.active:
            return []
        lowered = text.lower()
        # Preserve original Unicode code-point offsets, even for expanding lowercase characters.
        mapping = None
        if len(lowered) != len(text):
            mapping = [i for i, c in enumerate(text) for _ in c.lower()]
        found = []
        for end, (alias, ids) in self.automaton.iter(lowered):
            start = end - len(alias) + 1
            stop = end + 1
            if (start and lowered[start - 1].isalnum()) or (stop < len(lowered) and lowered[stop].isalnum()):
                continue
            a, b = (mapping[start], mapping[end] + 1) if mapping else (start, stop)
            for cid in ids:
                found.append({"concept_id": cid, "alias": text[a:b], "start": a, "end": b})
        return found


def discover(project: Project) -> dict:
    lock = read_json(project.path("taxonomy/candidates/ontology_lock.json"))
    path = project.path("taxonomy/candidates/doid.obo")
    if file_hash(path) != lock["sha256"]:
        raise PipelineError("Ontology checksum mismatch")
    terms = parse_obo(path.read_text(encoding="utf-8"), project.config("scope"))
    matcher = Matcher(terms)
    articles = {r["article_id"]: r for r in project.rows("data/normalized/articles.parquet")}
    counts, cases_count = Counter(), Counter()
    metadata_hits = {}
    for aid, article in articles.items():
        metadata = "\n".join([article["title"]] + article["keywords"] + article["mesh_terms"])
        metadata_hits[aid] = matcher.find(metadata)
    for case in project.rows("data/normalized/cases.parquet"):
        found = matcher.find(case["raw_case_text"]) + metadata_hits[case["article_id"]]
        counts.update(f["concept_id"] for f in found)
        cases_count.update({f["concept_id"] for f in found})
    for term in terms:
        term["raw_mentions"] = counts[term["concept_id"]]
        term["candidate_cases"] = cases_count[term["concept_id"]]
        term["status"] = "in_scope" if term["in_scope"] else "out_of_scope"
    project.write("data/catalog/disease_catalog.parquet", terms)
    return {"ontology_concepts": len(terms), "observed_concepts": len(counts),
            "observed_infectious_concepts": sum(r["in_scope"] and r["raw_mentions"] > 0 for r in terms)}

