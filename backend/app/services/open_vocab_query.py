from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any


PRIMARY_CLASSES = {"person", "car", "motorcycle", "bicycle", "bus", "truck"}

PRIMARY_ALIASES: dict[str, tuple[str, ...]] = {
    "person": ("person", "people", "pedestrian", "man", "woman", "nguoi", "nguoi di bo"),
    "car": ("car", "auto", "automobile", "o to", "oto", "xe hoi", "xe con"),
    "motorcycle": ("motorcycle", "motorbike", "scooter", "xe may", "mo to"),
    "bicycle": ("bicycle", "bike", "xe dap"),
    "bus": ("bus", "xe bus", "xe buyt"),
    "truck": ("truck", "xe tai"),
}

OPEN_VOCAB_ALIASES: dict[str, tuple[str, ...]] = {
    "dog": ("dog", "con cho", "cho"),
    "cat": ("cat", "con meo", "meo"),
    "ambulance": ("ambulance", "xe cuu thuong"),
    "fire truck": ("fire truck", "fire engine", "xe cuu hoa"),
    "traffic cone": ("traffic cone", "cone", "coc giao thong", "chop giao thong"),
    "umbrella": ("umbrella", "cai o", "du"),
    "suitcase": ("suitcase", "vali", "va li"),
    "chair": ("chair", "ghe"),
    "traffic sign": ("traffic sign", "road sign", "bien bao", "bien bao giao thong"),
}

COLOR_TERMS = {
    "red",
    "blue",
    "green",
    "white",
    "black",
    "yellow",
    "gray",
    "grey",
    "orange",
    "brown",
    "purple",
    "pink",
    "mau do",
    "mau xanh",
    "mau xanh duong",
    "mau xanh la",
    "mau trang",
    "mau den",
    "mau vang",
    "mau xam",
    "do",
    "xanh",
    "trang",
    "den",
    "vang",
    "xam",
}

RELATIONSHIP_TERMS = (
    "di xe",
    "dang di xe",
    "cuoi xe",
    "ngoi tren xe",
    "person on motorcycle",
    "person riding motorcycle",
    "riding",
)

QUERY_PREFIXES = (
    "tim kiem",
    "tim",
    "hay tim",
    "cho toi xem",
    "xem",
    "locate",
    "find",
    "search for",
    "search",
    "trong video co",
    "video co",
    "co",
)

QUERY_NOISE = {
    "a",
    "an",
    "the",
    "object",
    "objects",
    "video",
    "trong",
    "co",
    "khong",
    "hay",
    "tim",
    "kiem",
    "cho",
    "toi",
    "xem",
    "doan",
    "canh",
    "luc",
    "nao",
    "xuat",
    "hien",
    "con",
    "cai",
    "chiec",
    "mot",
    "nhung",
    "cac",
}


@dataclass(frozen=True)
class ObjectRoute:
    detector: str
    object_name: str | None
    status: str = "ok"
    message: str | None = None
    reason: str | None = None


def normalize_for_object_query(text: str) -> str:
    value = unicodedata.normalize("NFD", text.lower())
    value = value.replace("đ", "d")
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    value = re.sub(r"[^a-z0-9\s-]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def route_object_query(query: str, parsed_entities: dict[str, Any] | None = None) -> ObjectRoute:
    normalized = normalize_for_object_query(query)
    if not normalized:
        return ObjectRoute(
            detector="none",
            object_name=None,
            status="unsupported_query",
            message="Khong xac dinh duoc doi tuong can tim.",
            reason="empty_query",
        )

    if _has_color_attribute(normalized):
        return ObjectRoute(
            detector="none",
            object_name=None,
            status="unsupported_attribute",
            message="Chua du du lieu de xac minh thuoc tinh mau.",
            reason="color_attribute_requires_reliable_visual_verification",
        )

    if _has_relationship(normalized):
        return ObjectRoute(
            detector="none",
            object_name=None,
            status="unsupported_relationship",
            message="Truy van quan he phuc tap chua duoc ho tro day du.",
            reason="relationship_reasoning_not_enabled",
        )

    parsed_object = (parsed_entities or {}).get("object")
    if isinstance(parsed_object, str) and parsed_object in PRIMARY_CLASSES:
        return ObjectRoute(detector="primary", object_name=parsed_object)

    primary = _first_alias(normalized, PRIMARY_ALIASES)
    if primary:
        return ObjectRoute(detector="primary", object_name=primary)

    open_vocab = _first_alias(normalized, OPEN_VOCAB_ALIASES)
    if open_vocab:
        return ObjectRoute(detector="yolo_world", object_name=open_vocab)

    extracted = _extract_generic_object(normalized)
    if extracted is None:
        return ObjectRoute(
            detector="none",
            object_name=None,
            status="unsupported_query",
            message="Khong xac dinh duoc doi tuong can tim.",
            reason="object_extraction_failed",
        )

    return ObjectRoute(detector="yolo_world", object_name=extracted)


def looks_like_object_query(query: str) -> bool:
    normalized = normalize_for_object_query(query)
    if not normalized:
        return False
    if any(normalized.startswith(prefix) for prefix in QUERY_PREFIXES):
        return True
    if _first_alias(normalized, OPEN_VOCAB_ALIASES) is not None:
        return True
    return any(token in f" {normalized} " for token in (" luc nao ", " xuat hien ", " object "))


def build_cache_key(
    *,
    video_id: str,
    object_name: str,
    model: str,
    confidence: float,
    sample_fps: float,
) -> str:
    return "|".join(
        [
            str(video_id),
            sanitize_prompt(object_name),
            str(model),
            f"{float(confidence):.4f}",
            f"{float(sample_fps):.4f}",
        ]
    )


def sanitize_prompt(value: str, max_length: int = 80) -> str:
    normalized = normalize_for_object_query(value)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized[:max_length].strip()


def _first_alias(text: str, aliases: dict[str, tuple[str, ...]]) -> str | None:
    padded = f" {text} "
    matches: list[tuple[int, str]] = []
    for canonical, words in aliases.items():
        for word in words:
            needle = f" {normalize_for_object_query(word)} "
            index = padded.find(needle)
            if index >= 0:
                matches.append((index, canonical))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0])
    return matches[0][1]


def _has_color_attribute(text: str) -> bool:
    padded = f" {text} "
    return any(f" {term} " in padded for term in COLOR_TERMS)


def _has_relationship(text: str) -> bool:
    return any(term in text for term in RELATIONSHIP_TERMS)


def _extract_generic_object(text: str) -> str | None:
    candidate = text
    for prefix in QUERY_PREFIXES:
        if candidate.startswith(prefix + " "):
            candidate = candidate[len(prefix) + 1 :]
            break

    candidate = re.sub(r"\b(khong|co khong|trong video|xuat hien|luc nao|o dau)\b", " ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip(" -")

    words = [word for word in candidate.split() if word not in QUERY_NOISE]
    candidate = " ".join(words).strip()
    if not candidate or len(candidate) < 2:
        return None
    if len(candidate) > 60:
        return None
    if not re.fullmatch(r"[a-z0-9][a-z0-9\s-]*", candidate):
        return None
    if candidate in QUERY_NOISE:
        return None
    return candidate
