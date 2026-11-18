"""
Reference examples of the contract (SVC-12) pass the API schemas, extra fields forbidden.
"""

import json

from pathlib import Path

import pytest

from pydantic import BaseModel

from app.api_v1.schemas import (
    BenchResult,
    DetectResponse,
    ErrorResponse,
    InfoParams,
    InfoResponse,
    VocabularyResponse,
)

CONTRACT = Path(__file__).resolve().parents[1] / "tests" / "contract"

SCHEMAS = {
    "info.response.json": InfoResponse,
    "detect.response.json": DetectResponse,
    "vocabulary.response.json": VocabularyResponse,
    "error.response.json": ErrorResponse,
    "bench.result.json": BenchResult,
}


def test_contract_covers_every_example():
    # A new example without a schema would pass unchecked
    examples = {path.name for path in CONTRACT.glob("*.json") if not path.name.endswith(".cases.json")}
    assert examples == set(SCHEMAS)


@pytest.mark.parametrize("name", sorted(SCHEMAS))
def test_contract_example_matches_schema(name):
    schema = SCHEMAS[name]
    example = json.loads((CONTRACT / name).read_text())

    # extra="forbid" all the way down: a field of the example the schema lacks fails here
    assert _forbids_extra(schema), name
    parsed = schema.model_validate(example)
    assert parsed.model_dump(mode="json") == example


def test_contract_info_params():
    # The SVC-2 example has no unsupported_classes (DET-3) and a placeholder hash
    params = json.loads((CONTRACT / "info.response.json").read_text())["params"]
    parsed = InfoParams.model_validate(params)
    assert parsed.unsupported_classes is None
    assert (parsed.imgsz, parsed.conf) == (640, 0.35)


def test_contract_extra_field_fails():
    example = json.loads((CONTRACT / "detect.response.json").read_text())
    example["objects"][0]["depth"] = 1.2
    with pytest.raises(ValueError):
        DetectResponse.model_validate(example)


def _forbids_extra(schema: type) -> bool:
    if schema.model_config.get("extra") != "forbid":
        return False
    for field in schema.model_fields.values():
        for nested in _models_of(field.annotation):
            if not _forbids_extra(nested):
                return False
    return True


def _models_of(annotation) -> list:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    return [model for argument in getattr(annotation, "__args__", ()) for model in _models_of(argument)]
