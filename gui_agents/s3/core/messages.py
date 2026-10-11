"""Shared, dictionary-compatible types for multimodal LMM messages."""

from collections.abc import Sequence
from typing import Literal, TypedDict

MessageRole = Literal["system", "user", "assistant"]
ImageDetail = Literal["auto", "low", "high"]


class TextContent(TypedDict):
    type: Literal["text"]
    text: str


class ImageURLRequired(TypedDict):
    url: str


class ImageURL(ImageURLRequired, total=False):
    detail: ImageDetail


class ImageURLContent(TypedDict):
    type: Literal["image_url"]
    image_url: ImageURL


class Base64ImageSource(TypedDict):
    type: Literal["base64"]
    media_type: str
    data: str


class Base64ImageContent(TypedDict):
    type: Literal["image"]
    source: Base64ImageSource


MessageContent = TextContent | ImageURLContent | Base64ImageContent


class LMMMessage(TypedDict):
    role: MessageRole
    content: list[MessageContent]


class TextMessage(TypedDict):
    role: MessageRole
    content: str


def normalize_messages(
    messages: Sequence[LMMMessage | TextMessage],
) -> list[LMMMessage]:
    """Accept legacy string content at the boundary; keep internal blocks typed."""
    normalized: list[LMMMessage] = []
    for message in messages:
        content = message["content"]
        normalized.append(
            {
                "role": message["role"],
                "content": (
                    [{"type": "text", "text": content}]
                    if isinstance(content, str)
                    else list(content)
                ),
            }
        )
    return normalized
