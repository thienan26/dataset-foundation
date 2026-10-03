from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .common import PipelineError, Project


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="multicare-data", description="Curate a reproducible MultiCaRe dataset")
    root.add_argument("--root", default=".", help="Project root; source data is stored under this root")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("medgemma-export", help="Export source-bound local MedGemma jobs")
    mg_import = commands.add_parser("medgemma-import", help="Validate and import MedGemma JSONL results")
    mg_import.add_argument("input")
    mg_run = commands.add_parser("medgemma-run", help="Run resumable MedGemma reviews")
    mg_run.add_argument("--backend", choices=["local", "endpoint"], default="local")
    mg_run.add_argument("--limit", type=int, default=25)
    prehuman = commands.add_parser("prepare-human-review", help="Refresh pre-human artifacts and final review bundle")
    prehuman.add_argument("--allow-pending", action="store_true", help="Export a diagnostic draft when AI reviews are incomplete")
    commands.add_parser("resolve-latest", help="Compare the current Zenodo latest record with the project pin")
    check = commands.add_parser("check-gemini", help="Test Gemini credentials with a generic prompt and no dataset data")
    check.add_argument("--model", help="Optional model ID for a generic connectivity check")
    check.add_argument("--plain-text", action="store_true", help="Use a plain-text probe without a JSON schema")
    acquire = commands.add_parser("acquire", help="Download the exact pinned source to data/raw on the project drive")
    acquire.add_argument("--tables-only", action="store_true", help="Download normalized source tables before the image archives")
    verify = commands.add_parser("verify-source")
    verify.add_argument("--require-images", action="store_true")
    for command, help_text in [
        ("extract-images", "Safely extract the checksum-verified PMC source archives"),
        ("normalize", "Normalize the pinned source into article, case and image Parquet entities"),
        ("ontology", "Download and pin the selected Human Disease Ontology revision"),
        ("discover", "Discover and count disease concepts across case text and article metadata"),
        ("build-candidates", "Create high-recall infectious disease candidate hypotheses"),
        ("label", "Assign statuses, literal evidence offsets and single-label routing"),
        ("adjudicate", "Combine current rule, LLM and human decisions without overwriting them"),
        ("deduplicate", "Build article, text, pixel and near-image connected components"),
        ("feasibility", "Calculate usable class and independent-group support"),
        ("freeze-taxonomy", "Write the versioned taxonomy once feasibility passes"),
        ("split", "Create the group-based frozen test set and development folds"),
        ("evaluate", "Run the release gates and generate the quarantine artifact"),
    ]:
        commands.add_parser(command, help=help_text)
    for name, role in (("review-a", "a"), ("review-b", "b")):
        command = commands.add_parser(name, help=f"Run cached/rate-limited Gemini reviewer {role.upper()}")
        command.add_argument("--limit", type=int, help="Maximum new paid/API review calls")
    for name, module_name in (("text-qc", "text"), ("image-qc", "image")):
        command = commands.add_parser(name, help=f"Run deterministic {module_name} QC and optional Gemini review")
        command.add_argument("--with-ai", action="store_true")
        command.add_argument("--limit", type=int, default=25, help="Maximum new Gemini calls; cached calls cost zero")
    human_export = commands.add_parser("human-export", help="Prepare the predeclared human review queue")
    human_export.add_argument("--output", default="data/reviews/human_input.jsonl")
    pre_ai_export = commands.add_parser("export-pre-ai-review", help="Join all pre-Gemini single-target case artifacts into one Parquet file")
    pre_ai_export.add_argument("--output", default="data/reviews/pre_ai_review.parquet")
    human_import = commands.add_parser("human-import", help="Validate and append reviewed JSONL records")
    human_import.add_argument("input")
    adjudication_export = commands.add_parser("human-adjudication-export",
                                              help="Export unresolved reviewer A/B conflicts for label-only adjudication")
    adjudication_export.add_argument("--output", default="data/reviews/human_adjudication_input.jsonl")
    adjudication_import = commands.add_parser("human-adjudication-import",
                                              help="Validate and append label-only human adjudications")
    adjudication_import.add_argument("input")
    freeze_cmd = commands.add_parser("freeze", help="Create a checksummed, write-once dataset release")
    freeze_cmd.add_argument("--version", default="dataset_v001")
    verify_release = commands.add_parser("verify-release")
    verify_release.add_argument("version", nargs="?", default="dataset_v001")
    return root


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    project = Project(args.root)
    try:
        command = args.command
        if command.startswith("medgemma-"):
            from .medgemma import export_jobs, import_results, run, run_local
            if command == "medgemma-export":
                result = export_jobs(project)
            elif command == "medgemma-import":
                result = import_results(project, args.input)
            elif args.limit <= 0:
                raise PipelineError("--limit must be positive")
            else:
                result = run_local(project, args.limit) if args.backend == "local" else run(project, args.limit)
        elif command == "prepare-human-review":
            from .final_review import prepare
            result = prepare(project, args.allow_pending)
        elif command == "check-gemini":
            from .provider_check import check_gemini
            result = check_gemini(project, args.model, args.plain_text)
        elif command in {"resolve-latest", "acquire", "verify-source", "extract-images", "normalize"}:
            from .acquire import acquire, extract_images, resolve_latest, verify_source
            if command == "resolve-latest":
                from .common import write_json
                result = resolve_latest(project)
                write_json(project.path("reports/latest_source_check.json"), result)
                result["recommendation"] = "pin already-current" if result["matches_pin"] else "review newer release and deliberately update configs/source.yaml"
            elif command == "acquire":
                result = acquire(project, tables_only=args.tables_only)
            elif command == "verify-source":
                result = verify_source(project, require_images=args.require_images)
            elif command == "extract-images":
                result = extract_images(project)
            else:
                from .normalize import normalize
                result = normalize(project)
        elif command in {"ontology", "discover"}:
            from .discover import acquire_ontology, discover
            result = acquire_ontology(project) if command == "ontology" else discover(project)
        elif command == "build-candidates":
            from .candidates import build_candidates
            result = build_candidates(project)
        elif command == "label":
            from .labeling import label
            result = label(project)
        elif command == "adjudicate":
            from .adjudicate import adjudicate
            result = adjudicate(project)
        elif command == "review-a" or command == "review-b":
            from .llm_review import review
            cfg = project.config("review")
            result = review(project, "a" if command == "review-a" else "b", args.limit or cfg["default_limit"])
        elif command == "text-qc":
            from .text_qc import text_qc
            result = text_qc(project, args.with_ai, args.limit)
        elif command == "image-qc":
            from .image_qc import image_qc
            result = image_qc(project, args.with_ai, args.limit)
        elif command == "deduplicate":
            from .grouping import group
            result = group(project)
        elif command == "feasibility":
            from .grouping import feasibility
            result = feasibility(project)
        elif command == "freeze-taxonomy":
            from .release import freeze_taxonomy
            result = freeze_taxonomy(project)
        elif command == "split":
            from .split import grouped_split
            result = grouped_split(project)
        elif command == "human-export":
            from .adjudicate import prepare_human
            path = Path(args.output)
            path = path if path.is_absolute() else project.path(str(path))
            result = prepare_human(project, path)
        elif command == "export-pre-ai-review":
            from .pre_review import export_pre_ai_review
            result = export_pre_ai_review(project, args.output)
        elif command == "human-import":
            from .adjudicate import import_human
            input_path = Path(args.input)
            if not input_path.is_absolute():
                input_path = project.path(str(input_path))
            result = import_human(project, input_path)
        elif command == "human-adjudication-export":
            from .human_adjudication import export_human_adjudication
            result = export_human_adjudication(project, args.output)
        elif command == "human-adjudication-import":
            from .human_adjudication import import_human_adjudication
            result = import_human_adjudication(project, args.input)
        elif command == "evaluate":
            from .adjudicate import adjudicate
            from .evaluate import evaluate
            reviews = adjudicate(project)
            result = {"adjudication": reviews, "release_gate": evaluate(project)}
        elif command == "freeze":
            from .release import freeze
            result = freeze(project, args.version)
        elif command == "verify-release":
            from .release import verify_release
            result = verify_release(project, args.version)
        else:
            raise PipelineError(f"Unknown command: {command}")
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except (PipelineError, FileNotFoundError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
