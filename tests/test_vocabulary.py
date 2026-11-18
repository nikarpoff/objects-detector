"""
Class vocabulary (DET-2, DET-3): checked at load, mentions work in Python, the API gives the contract.
"""

import copy
import json
import re

from pathlib import Path

import pytest
import yaml

from app.core.detection.mapping import ClassMapping
from app.core.detection.vocabulary import (
    REQUIRED_CONTEXT_CLASSES,
    REQUIRED_KEY_CLASSES,
    VocabularyError,
    load_vocabulary,
    parse_vocabulary,
)

ROOT = Path(__file__).resolve().parents[1]
VOCABULARY_PATH = ROOT / "configs" / "vocabulary.yaml"
CONTRACT = ROOT / "tests" / "contract" / "vocabulary.response.json"


@pytest.fixture(scope="module")
def document() -> dict:
    return yaml.safe_load(VOCABULARY_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def vocabulary():
    return load_vocabulary(VOCABULARY_PATH)


def test_vocabulary_required_classes(vocabulary):
    assert len(vocabulary.classes) == 26
    assert vocabulary.key_ids == frozenset(REQUIRED_KEY_CLASSES)
    assert set(vocabulary.ids) == set(REQUIRED_KEY_CLASSES + REQUIRED_CONTEXT_CLASSES)
    assert len(set(vocabulary.ids)) == len(vocabulary.ids)


def test_vocabulary_forms(vocabulary):
    for item in vocabulary.classes:
        assert re.fullmatch(r"[a-z][a-z0-9_]*", item.id)
        assert item.ru.one and item.ru.few and item.ru.many, item.id
        assert item.ru.gender in ("m", "f", "n", "pl"), item.id
        assert item.prompts, item.id

    chair = vocabulary.get("chair").ru
    assert (chair.one, chair.few, chair.many, chair.gender) == ("стул", "стула", "стульев", "m")


def test_vocabulary_hash_matches_contract(vocabulary):
    contract = json.loads(CONTRACT.read_text())
    assert vocabulary.hash == contract["hash"] == "8778f20c71d0"
    assert vocabulary.document == contract["classes"]


def test_vocabulary_mentions_match_word_forms(vocabulary):
    # Every form of a name is a mention of its class and of no other one
    for item in vocabulary.classes:
        for form in (item.ru.one, item.ru.few, item.ru.many):
            text = form.lower().replace("ё", "е")
            assert vocabulary.mentions(text) == [item.id], form


@pytest.mark.parametrize(
    "text, expected",
    [
        ("слева дверь впереди стол", ["door", "table"]),
        ("справа две двери", ["door"]),
        ("впереди лестница и столб", ["stairs", "pole"]),
        ("рядом стулья и скамейка", ["bench", "chair"]),
        ("на переходе люди", ["person", "crosswalk"]),
        ("впереди электросамокат", ["scooter"]),
        ("столбик у двери", ["door", "pole"]),
        ("на столе ничего", ["table"]),
        ("пешеходный тротуар", []),
        ("дверной проем", []),
    ],
)
def test_vocabulary_mentions_in_answers(vocabulary, text, expected):
    assert vocabulary.mentions(text) == [class_id for class_id in vocabulary.ids if class_id in expected]


def test_vocabulary_mentions_compile_in_python(document):
    for entry in document["classes"]:
        for mention in entry["mentions"]:
            re.compile(f"(?:^| )(?:{mention})(?: |$)")
            assert "\\" not in mention and "ё" not in mention


def broken(document: dict, change) -> dict:
    document = copy.deepcopy(document)
    change(document["classes"])
    return document


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda classes: classes.pop(1), "required class 'door' is missing"),
        (lambda classes: classes.append(copy.deepcopy(classes[0])), "duplicate class 'person'"),
        (lambda classes: classes[0].update(id="Person"), "class #1: invalid id 'Person', lowercase latin snake_case is required"),
        (lambda classes: classes[0]["ru"].pop("few"), "class 'person': ru must have exactly one, few, many and gender"),
        (lambda classes: classes[0]["ru"].update(many=" "), "class 'person': form 'many' is empty"),
        (lambda classes: classes[0]["ru"].update(gender="x"), "class 'person': unknown gender 'x', expected one of: f, m, n, pl"),
        (lambda classes: classes[0].update(key="yes"), "class 'person': key must be true or false"),
        (lambda classes: classes[16].update(key=True), "class 'bed' must have key: false"),
        (lambda classes: classes[0].pop("mentions"), "class 'person': missing fields mentions"),
        (lambda classes: classes[0].update(color="red"), "class 'person': unknown fields color"),
        (lambda classes: classes[0].update(prompts=[]), "class 'person': prompts must be a non-empty list"),
        (lambda classes: classes[0].update(prompts=["Человек"]), "class 'person': invalid prompt 'Человек', lowercase english words are required"),
        (lambda classes: classes[1].update(prompts=["person"]), "class 'door': prompt 'person' is taken by class 'person'"),
        (lambda classes: classes[0].update(mentions=["человек("]), "class 'person': mention 'человек(' does not compile"),
        (lambda classes: classes[0].update(mentions=["\\bчеловек"]), "class 'person': mention '\\bчеловек' uses escapes, lookaround or backreferences"),
        (lambda classes: classes[0].update(mentions=["чел(?=овек)"]), "class 'person': mention 'чел(?=овек)' uses escapes, lookaround or backreferences"),
        (lambda classes: classes[0].update(mentions=["ребёнок"]), "class 'person': mention 'ребёнок' must be lowercase with «е» for «ё»"),
    ],
    ids=[
        "missing_required", "duplicate_id", "bad_id", "missing_form", "empty_form", "bad_gender", "bad_key",
        "context_as_key", "missing_field", "unknown_field", "no_prompts", "russian_prompt", "shared_prompt",
        "mention_not_compiling", "mention_word_boundary", "mention_lookahead", "mention_yo",
    ],
)
def test_vocabulary_invalid_document_fails(document, change, message):
    with pytest.raises(VocabularyError, match=re.escape(message)):
        parse_vocabulary(broken(document, change))


@pytest.mark.parametrize(
    "content, message",
    [
        ("classes: [person", "is not valid YAML"),
        ("classes: []", "a non-empty list 'classes' is required"),
        ("- person", "a non-empty list 'classes' is required"),
    ],
    ids=["bad_yaml", "no_classes", "not_mapping"],
)
def test_vocabulary_broken_file_fails(tmp_path, content, message):
    path = tmp_path / "vocabulary.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(VocabularyError, match=re.escape(f"Vocabulary '{path}'")) as error:
        load_vocabulary(path)
    assert message in str(error.value)


def test_vocabulary_missing_file_fails(tmp_path):
    path = tmp_path / "missing.yaml"
    with pytest.raises(VocabularyError, match=re.escape(f"Vocabulary '{path}' cannot be read")):
        load_vocabulary(path)


def test_vocabulary_select(vocabulary):
    assert vocabulary.select(None) is None
    assert vocabulary.select(" , ") is None
    assert vocabulary.select("door, chair") == frozenset({"door", "chair"})


def test_vocabulary_endpoint_matches_contract(client, auth):
    # The same check as tests/test_api.py::test_vocabulary, kept with the vocabulary rules
    assert client.get("/v1/vocabulary", headers=auth).json() == json.loads(CONTRACT.read_text())


def test_mapping_closed_vocabulary(vocabulary):
    # COCO names of a closed-vocabulary model: what is not in COCO is unsupported (DET-3)
    coco = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "bench", "dog",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "sink", "refrigerator"]
    mapping = ClassMapping(vocabulary, coco)

    assert mapping.class_id("dining table") == "table"
    assert mapping.class_id("Traffic_Light") == "traffic_light"
    assert mapping.class_id("couch") == "sofa"
    assert mapping.class_id("potted plant") is None
    assert mapping.dropped == ("potted plant", )
    assert set(mapping.unsupported) == {"door", "stairs", "crosswalk", "pole", "barrier", "wardrobe", "window", "scooter", "trash_can"}


def test_mapping_open_vocabulary(vocabulary):
    prompts = [prompt for item in vocabulary.classes for prompt in item.prompts]
    mapping = ClassMapping(vocabulary, prompts)

    assert mapping.unsupported == ()
    assert mapping.class_id("zebra crossing") == "crosswalk"
    assert mapping.class_id("staircase") == "stairs"
