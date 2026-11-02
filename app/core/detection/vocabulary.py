"""
Class vocabulary (DET-2): one source of the class names for the detector, the gateway
templates and the evaluation.

The vocabulary is a contract shared with the gateway: the gateway compiles the mentions
with RE2 and compares the hash. So a broken file stops the start with a clear message
instead of answering with classes the gateway cannot name.
"""

import hashlib
import json
import re

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import yaml

from app.core.exceptions import BackendError, InvalidParamsError

HASH_LENGTH = 12                    # hex digits of SHA-256 in the hash

# A mention matches whole words of the normalized text: RE2 has no Unicode \b
MENTION_FORMAT = "(?:^| )(?:{})(?: |$)"

# Key classes: the hypothesis metric is counted over them
REQUIRED_KEY_CLASSES = (
    "person", "door", "stairs", "car", "bus", "truck", "bicycle", "motorcycle",
    "scooter", "traffic_light", "crosswalk", "pole", "bench", "barrier", "chair", "table",
)
REQUIRED_CONTEXT_CLASSES = (
    "bed", "wardrobe", "sofa", "window", "sink", "toilet", "refrigerator", "tv", "dog", "trash_can",
)

GENDERS = ("f", "m", "n", "pl")
CLASS_FIELDS = frozenset({"id", "prompts", "ru", "key", "mentions"})
FORM_FIELDS = frozenset({"one", "few", "many", "gender"})

_CLASS_ID = re.compile(r"[a-z][a-z0-9_]*")
_PROMPT = re.compile(r"[a-z0-9]+(?:[ -][a-z0-9]+)*")
# Out of the subset common to RE2 and re: escapes (\b among them), lookaround, backreferences
_NOT_PORTABLE = re.compile(r"\\|\(\?[=!<]|\(\?P=")


class VocabularyError(BackendError):
    """The vocabulary file is missing or breaks DET-2: the backend cannot be built without it."""


@dataclass(frozen=True)
class RussianForms:
    """Russian name of a class: «один стул», «два стула», «пять стульев»"""
    one: str
    few: str
    many: str
    gender: str


@dataclass(frozen=True)
class VocabularyClass:
    """A class of the vocabulary"""
    id: str
    prompts: Tuple[str, ...]
    ru: RussianForms
    key: bool
    mentions: Tuple[str, ...]
    patterns: Tuple[re.Pattern, ...] = field(default=(), repr=False, compare=False)

    def mentioned(self, text: str) -> bool:
        """Whether the text, normalized by ASR-4, mentions the class."""
        return any(pattern.search(text) for pattern in self.patterns)


class Vocabulary:
    """A checked vocabulary: the classes in the file order and the hash of the contract"""

    def __init__(self, classes: List[Dict[str, Any]]) -> None:
        # The document as loaded: /v1/vocabulary gives it back and the hash is taken of it
        self.document = classes
        self.hash = vocabulary_hash(classes)
        self.classes = tuple(_build_class(entry) for entry in classes)
        self._index = {vocabulary_class.id: vocabulary_class for vocabulary_class in self.classes}

    @property
    def ids(self) -> Tuple[str, ...]:
        return tuple(self._index)

    @property
    def key_ids(self) -> FrozenSet[str]:
        return frozenset(vocabulary_class.id for vocabulary_class in self.classes if vocabulary_class.key)

    def get(self, class_id: str) -> VocabularyClass:
        return self._index[class_id]

    def __contains__(self, class_id: object) -> bool:
        return class_id in self._index

    def mentions(self, text: str) -> List[str]:
        """Ids of the classes the normalized text mentions, in the vocabulary order."""
        return [vocabulary_class.id for vocabulary_class in self.classes if vocabulary_class.mentioned(text)]

    def select(self, classes: Optional[str]) -> Optional[FrozenSet[str]]:
        """Ids of the `classes` query parameter (comma separated); None - all of them."""
        ids = [class_id.strip() for class_id in (classes or "").split(",") if class_id.strip()]
        if not ids:
            return None

        unknown = [class_id for class_id in ids if class_id not in self._index]
        if unknown:
            raise InvalidParamsError(f"Unknown class '{unknown[0]}' in classes")
        return frozenset(ids)


def vocabulary_hash(classes: List[Dict[str, Any]]) -> str:
    """First 12 hex of SHA-256 of the canonical JSON of the classes, the same in the gateway."""
    canonical = json.dumps(classes, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:HASH_LENGTH]


def load_vocabulary(path: Path) -> Vocabulary:
    """The vocabulary of the file, checked; VocabularyError names the file and the problem."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise VocabularyError(f"Vocabulary '{path}' cannot be read: {e.strerror or e}") from e

    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise VocabularyError(f"Vocabulary '{path}' is not valid YAML: {' '.join(str(e).split())}") from e

    try:
        return parse_vocabulary(document)
    except VocabularyError as e:
        raise VocabularyError(f"Vocabulary '{path}': {e}") from None


def parse_vocabulary(document: Any) -> Vocabulary:
    """The vocabulary of a loaded document, checked against DET-2."""
    if not isinstance(document, dict) or not isinstance(document.get("classes"), list) or not document["classes"]:
        raise VocabularyError("a non-empty list 'classes' is required")

    classes = document["classes"]
    seen_ids, seen_prompts = set(), {}
    for position, entry in enumerate(classes, start=1):
        _check_class(entry, position)

        class_id = entry["id"]
        if class_id in seen_ids:
            raise VocabularyError(f"duplicate class '{class_id}'")
        seen_ids.add(class_id)

        # A prompt of two classes would make the mapping of the model classes ambiguous (DET-3)
        for prompt in entry["prompts"]:
            if prompt in seen_prompts:
                raise VocabularyError(f"class '{class_id}': prompt '{prompt}' is taken by class '{seen_prompts[prompt]}'")
            seen_prompts[prompt] = class_id

    by_id = {entry["id"]: entry for entry in classes}
    for class_id in REQUIRED_KEY_CLASSES + REQUIRED_CONTEXT_CLASSES:
        if class_id not in by_id:
            raise VocabularyError(f"required class '{class_id}' is missing")
        is_key = class_id in REQUIRED_KEY_CLASSES
        if by_id[class_id]["key"] is not is_key:
            raise VocabularyError(f"class '{class_id}' must have key: {str(is_key).lower()}")

    return Vocabulary(classes)


def _check_class(entry: Any, position: int) -> None:
    if not isinstance(entry, dict):
        raise VocabularyError(f"class #{position} is not a mapping")

    class_id = entry.get("id")
    if not isinstance(class_id, str) or not _CLASS_ID.fullmatch(class_id):
        raise VocabularyError(f"class #{position}: invalid id {class_id!r}, lowercase latin snake_case is required")

    fields = set(entry)
    if fields != CLASS_FIELDS:
        missing, unknown = sorted(CLASS_FIELDS - fields), sorted(fields - CLASS_FIELDS)
        problem = f"missing fields {', '.join(missing)}" if missing else f"unknown fields {', '.join(unknown)}"
        raise VocabularyError(f"class '{class_id}': {problem}")

    prompts = entry["prompts"]
    if not isinstance(prompts, list) or not prompts:
        raise VocabularyError(f"class '{class_id}': prompts must be a non-empty list")
    for prompt in prompts:
        if not isinstance(prompt, str) or not _PROMPT.fullmatch(prompt):
            raise VocabularyError(f"class '{class_id}': invalid prompt {prompt!r}, lowercase english words are required")

    forms = entry["ru"]
    if not isinstance(forms, dict) or set(forms) != FORM_FIELDS:
        raise VocabularyError(f"class '{class_id}': ru must have exactly one, few, many and gender")
    for name in ("one", "few", "many"):
        if not isinstance(forms[name], str) or not forms[name].strip():
            raise VocabularyError(f"class '{class_id}': form '{name}' is empty")
    if forms["gender"] not in GENDERS:
        raise VocabularyError(f"class '{class_id}': unknown gender {forms['gender']!r}, expected one of: {', '.join(GENDERS)}")

    if not isinstance(entry["key"], bool):
        raise VocabularyError(f"class '{class_id}': key must be true or false")

    mentions = entry["mentions"]
    if not isinstance(mentions, list) or not mentions:
        raise VocabularyError(f"class '{class_id}': mentions must be a non-empty list")
    for mention in mentions:
        _check_mention(class_id, mention)


def _check_mention(class_id: str, mention: Any) -> None:
    if not isinstance(mention, str) or not mention:
        raise VocabularyError(f"class '{class_id}': mention {mention!r} is not a pattern")
    # The text is normalized before the search (ASR-4): other patterns would never match
    if _NOT_PORTABLE.search(mention):
        raise VocabularyError(f"class '{class_id}': mention '{mention}' uses escapes, lookaround or backreferences")
    if mention != mention.lower() or "ё" in mention:
        raise VocabularyError(f"class '{class_id}': mention '{mention}' must be lowercase with «е» for «ё»")
    try:
        re.compile(MENTION_FORMAT.format(mention))
    except re.error as e:
        raise VocabularyError(f"class '{class_id}': mention '{mention}' does not compile: {e}") from e


def _build_class(entry: Dict[str, Any]) -> VocabularyClass:
    forms = entry["ru"]
    return VocabularyClass(
        id=entry["id"],
        prompts=tuple(entry["prompts"]),
        ru=RussianForms(one=forms["one"], few=forms["few"], many=forms["many"], gender=forms["gender"]),
        key=entry["key"],
        mentions=tuple(entry["mentions"]),
        patterns=tuple(re.compile(MENTION_FORMAT.format(mention)) for mention in entry["mentions"]),
    )
