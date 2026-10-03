STATUSES = ["confirmed", "probable", "suspected", "differential", "ruled_out", "historical", "unclear"]


def object_schema(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": required or list(properties), "additionalProperties": False}


LABEL_REVIEW = object_schema({
    "diagnosis_status": {"type": "string", "enum": STATUSES},
    "disease_is_current": {"type": "boolean"},
    "evidence_quote": {"type": "string"},
    "multiple_target_diseases": {"type": "boolean"},
    "other_target_diseases": {"type": "array", "items": {"type": "string"}},
    "decision": {"type": "string", "enum": ["accept", "reject", "uncertain"]},
    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    "reason_codes": {"type": "array", "items": {"type": "string"}},
    "reason": {"type": "string"},
})
TEXT_REVIEW = object_schema({
    "has_direct_leakage": {"type": "boolean"},
    "leaking_span": {"type": "string"},
    "usable_prediagnosis_context": {"type": "boolean"},
    "reason": {"type": "string"},
})
IMAGE_REVIEW = object_schema({
    "image_relevance": {"type": "string", "enum": ["diagnostic", "supportive", "irrelevant", "post_treatment", "unclear"]},
    "visible_diagnosis_text": {"type": "boolean"},
    "case_image_consistent": {"type": "boolean"},
    "reason": {"type": "string"},
})
HUMAN_REVIEW = object_schema({
    "case_id": {"type": "string", "minLength": 1},
    "candidate_disease": {"type": "string", "minLength": 1},
    "raw_text_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    "human_status": {"type": "string", "enum": STATUSES},
    "human_decision": {"type": "string", "enum": ["accept", "reject", "uncertain"]},
    "human_evidence_valid": {"type": "boolean"},
    "evidence_quote": {"type": "string"},
    "multiple_target_diseases": {"type": "boolean"},
    "text_leakage_found": {"type": "boolean"},
    "clinical_text_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    "selected_image_id": {"type": "string", "minLength": 1},
    "image_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    "image_relevance": {"type": "string", "enum": ["diagnostic", "supportive", "irrelevant", "post_treatment", "unclear"]},
    "image_label_leakage": {"type": "boolean"},
    "reason_code": {"type": "string", "minLength": 1},
    "reviewer_id": {"type": "string", "minLength": 1},
    "reviewed_at": {"type": "string", "format": "date-time"},
})

