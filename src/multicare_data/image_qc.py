from __future__ import annotations

import hashlib
import io
import warnings
from collections import Counter

import imagehash
from PIL import Image, ImageOps

from .common import Project, file_hash
from .llm_review import Gemini, review_key
from .review_schema import IMAGE_REVIEW
from .sample import single_targets


def inspect_image(path, cfg):
    if not path.is_file():
        return {"decode_pass": False, "reason": "IMAGE_MISSING"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                width, height = image.size
                if min(width, height) < cfg["minimum_image_side"] or width * height > cfg["maximum_image_pixels"]:
                    return {"decode_pass": False, "reason": "IMAGE_DIMENSIONS_INVALID"}
                image.verify()
            with Image.open(path) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                rgb.load()
                pixel_hash = hashlib.sha256()
                pixel_hash.update(f"RGB:{rgb.width}x{rgb.height}:".encode())
                pixel_hash.update(rgb.tobytes())
                pixel = pixel_hash.hexdigest()
                phash = str(imagehash.phash(rgb))
        return {"decode_pass": True, "reason": "PASS", "width": width, "height": height,
                "image_sha256": file_hash(path), "pixel_sha256": pixel, "perceptual_hash": phash}
    except MemoryError:
        return {"decode_pass": False, "reason": "IMAGE_MEMORY_LIMIT"}
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        return {"decode_pass": False, "reason": "IMAGE_INVALID"}


def image_payload(row, image):
    return {"task": "Audit the attached medical image for relevance to this case and visible diagnosis text. Inspect the actual pixels; captions alone are insufficient.",
            "target_disease": row["disease"], "raw_case_text": row["raw_case_text"],
            "caption": image["caption"],
            "instructions": "Classify diagnostic/supportive/irrelevant/post_treatment/unclear. Flag all visible text directly naming the target or its pathogen. Do not accept post-treatment, diagrams/charts unrelated to the patient's diagnosis or ambiguous case linkage."}


def review_image_bytes(path):
    # Downsample only the API copy; source bytes and all source checksums remain untouched.
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((3072, 3072), Image.Resampling.LANCZOS)
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=88, optimize=True)
        return stream.getvalue()


def selection_key(row, cfg):
    modality = row["image_modality"]
    modalities = cfg["modality_preference"]
    return (0 if row["image_relevance"] == "diagnostic" else 1,
            modalities.index(modality) if modality in modalities else len(modalities),
            0 if row["caption"] else 1, -row["width"] * row["height"], row["image_id"])


def image_qc(project: Project, use_ai=False, limit=25):
    cases = {r["case_id"]: r for r in single_targets(project)}
    previous = {r["image_id"]: r for r in project.rows("data/qc/image_qc.parquet", optional=True)}
    cfg, rcfg = project.config("qc"), project.config("review")
    client = Gemini(project) if use_ai else None
    rows, requests = [], 0
    for image in project.iter_rows("data/normalized/images.parquet"):
        row = cases.get(image["case_id"])
        if not row:
            continue
        path = project.path(image["path"])
        qc = {**image, **inspect_image(path, cfg)}
        result, key = None, None
        if qc["decode_pass"]:
            payload = image_payload(row, image)
            key = review_key("image_qc", payload, rcfg["qc_model"], rcfg, qc["image_sha256"])
            old = previous.get(image["image_id"], {})
            if old.get("ai_request_sha256") == key:
                result = {k: old[k] for k in ["image_relevance", "visible_diagnosis_text", "case_image_consistent", "ai_reason"]}
            elif client and requests < limit:
                saved = client.run("image_qc", payload, IMAGE_REVIEW, rcfg["qc_model"], path, qc["image_sha256"])
                result = dict(saved["result"])
                result["ai_reason"] = result.pop("reason")
                requests += 1
        for field in ["width", "height", "image_sha256", "pixel_sha256", "perceptual_hash"]:
            qc.setdefault(field, None)
        qc.update({"ai_request_sha256": key if result else None,
                   "image_relevance": result["image_relevance"] if result else "unclear",
                   "visible_diagnosis_text": result["visible_diagnosis_text"] if result else None,
                   "case_image_consistent": result["case_image_consistent"] if result else None,
                   "ai_reason": result["ai_reason"] if result else None})
        qc["image_usable"] = bool(qc["decode_pass"] and result
                                  and result["image_relevance"] in {"diagnostic", "supportive"}
                                  and not result["visible_diagnosis_text"] and result["case_image_consistent"])
        if qc["decode_pass"] and not qc["image_usable"]:
            qc["reason"] = "IMAGE_AI_REVIEW_REQUIRED" if not result else "IMAGE_AI_REJECTED"
        rows.append(qc)
    project.write("data/qc/image_qc.parquet", rows)
    selected = {}
    for row in sorted([r for r in rows if r["image_usable"]], key=lambda r: selection_key(r, cfg)):
        selected.setdefault(row["case_id"], row)
    project.write("data/qc/selected_images.parquet", list(selected.values()))
    return {"new_ai_reviews": requests, "selected_cases": len(selected),
            "counts": dict(Counter(r["reason"] for r in rows))}
