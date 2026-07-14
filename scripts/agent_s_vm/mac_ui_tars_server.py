#!/usr/bin/env python3
"""Minimal serial OpenAI-compatible server for private Agent S UI-TARS."""

from __future__ import annotations

import argparse
import base64
import time
import uuid
from io import BytesIO
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class ChatRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    max_tokens: int = 80
    temperature: float = 0.0


def _request_parts(messages: list[dict[str, Any]]) -> tuple[str, bytes]:
    text: list[str] = []
    images: list[bytes] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            text.append(content)
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                text.append(part["text"])
            if part.get("type") == "image_url":
                image_url = part.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else None
                if not isinstance(url, str) or not url.startswith(
                    "data:image/png;base64,"
                ):
                    raise HTTPException(422, "only inline PNG image_url input is allowed")
                try:
                    images.append(base64.b64decode(url.split(",", 1)[1], validate=True))
                except ValueError as exc:
                    raise HTTPException(422, "invalid inline PNG") from exc
    if len(images) != 1:
        raise HTTPException(422, "exactly one image is required")
    prompt = "\n".join(value.strip() for value in text if value.strip())
    if not prompt:
        raise HTTPException(422, "text prompt is required")
    return prompt, images[0]


def create_app(model_id: str) -> FastAPI:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="mps",
        local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        model_id, local_files_only=True, use_fast=False
    )
    app = FastAPI(title="Agent S private UI-TARS", version="0.2")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "healthy",
            "model": model_id,
            "runtime": "transformers-mps",
            "serial": True,
        }

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": model_id, "object": "model", "created": 0}],
        }

    @app.post("/v1/chat/completions")
    async def chat(request: ChatRequest) -> dict[str, Any]:
        prompt, png = _request_parts(request.messages)
        try:
            image = Image.open(BytesIO(png)).convert("RGB")
        except OSError as exc:
            raise HTTPException(422, "invalid PNG image") from exc
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        formatted = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[formatted], images=[image], return_tensors="pt"
        ).to("mps")
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max(1, min(request.max_tokens, 256)),
                do_sample=False,
            )
        generated = output_ids[:, inputs.input_ids.shape[1] :]
        content = processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ByteDance-Seed/UI-TARS-1.5-7B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()
    if args.host != "127.0.0.1":
        raise SystemExit("Agent S UI-TARS must bind to 127.0.0.1")
    uvicorn.run(create_app(args.model), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
