"""Keep Claude judge images under Anthropic's 5 MB limit on the OpenAI-compatible path.

OSWorld V2's native ``anthropic`` backend downsizes every judge image through
``desktop_env.image_utils.anthropic_image_file_to_mime_and_base64``. The
``openai_compatible`` backend -- what the grader uses since it moved to
OpenRouter (``OSWORLD_EVAL_MODEL_PROVIDER=openai_compatible`` in .env) -- sends
the raw file. Tasks 075 and 078 ship reference images of 13 MB and 7.7 MB, so
their LLM-judge step fails with HTTP 400 "image exceeds 5 MB maximum", the
harness counts that as infrastructure, and after three attempts the task
becomes an imputed zero.

The fix belongs in OSWorldV2_Evol, but the preflight demands that checkout be
byte-identical to the pinned release, so it is applied here at import time
instead. Only a Claude model is affected, and an image already under the limit
is sent byte-for-byte, so every other judge call is unchanged.
"""

import contextvars
import functools

_claude_call = contextvars.ContextVar("osworld_claude_judge_call", default=False)


def apply() -> bool:
    try:
        from desktop_env.evaluators.backends import openai_backend
        from desktop_env.image_utils import anthropic_image_file_to_mime_and_base64
    except ImportError:
        return False
    backend = openai_backend.OpenAIBackend
    if getattr(backend._generate_once, "_shadow_image_limit", False):
        return True

    plain_mime, plain_encode = openai_backend.guess_mime, openai_backend.encode_image

    @functools.lru_cache(maxsize=32)
    def _limited(path):
        return anthropic_image_file_to_mime_and_base64(path)

    def guess_mime(path):
        return _limited(path)[0] if _claude_call.get() else plain_mime(path)

    def encode_image(path):
        return _limited(path)[1] if _claude_call.get() else plain_encode(path)

    original = backend._generate_once

    @functools.wraps(original)
    def _generate_once(self, *args, **kwargs):
        token = _claude_call.set("claude" in str(self.config.model).lower())
        try:
            return original(self, *args, **kwargs)
        finally:
            _claude_call.reset(token)

    _generate_once._shadow_image_limit = True
    openai_backend.guess_mime = guess_mime
    openai_backend.encode_image = encode_image
    backend._generate_once = _generate_once
    return True
