"""GUI interaction primitives."""
import subprocess, urllib.request, json, base64, os

def take_screenshot(path="/tmp/screenshot.png"):
    """Capture the current screen to `path` (PNG, via scrot). Returns the path."""
    subprocess.run(["scrot", path], check=True)
    return path

def _grounding_provider():
    """Resolve which model grounding should call, from the agent's own env.

    Mirrors how the harness authenticates the agent, so grounding never runs on
    a different model than the island under test. Three real shapes:
    an OpenAI-compatible deployment, OpenRouter (OpenAI-compatible wire format
    on an ANTHROPIC_BASE_URL), and the native Anthropic API.
    """
    # The evaluator injects the exact task-agent model. This explicit binding
    # must win over ambient credentials: an OpenRouter Prime run may also carry
    # OPENAI_* for other repository services, and credential-order inference
    # would silently ground with the wrong deployment.
    selected_model = os.environ.get("SHADOW_GROUNDING_MODEL", "").strip()
    if selected_model.startswith("openrouter/"):
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is required for grounding")
        base = os.environ.get(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1",
        ).rstrip("/")
        return (
            "openai", base + "/chat/completions", key,
            selected_model.removeprefix("openrouter/"),
        )
    if selected_model.startswith("openai/"):
        base = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
        key = os.environ.get("OPENAI_API_KEY", "")
        if not base or not key:
            raise RuntimeError(
                "OPENAI_BASE_URL/OPENAI_API_KEY are required for grounding"
            )
        return (
            "openai", base + "/chat/completions", key,
            selected_model.removeprefix("openai/"),
        )
    if selected_model.startswith("anthropic/"):
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for grounding")
        base = os.environ.get(
            "ANTHROPIC_BASE_URL", "https://api.anthropic.com",
        ).rstrip("/")
        return (
            "anthropic", base + "/v1/messages", key,
            selected_model.removeprefix("anthropic/"),
        )

    openai_base = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    openai_model = os.environ.get("OPENAI_MODEL", "")
    if openai_base and openai_key and openai_model:
        return ("openai", openai_base + "/chat/completions", openai_key,
                openai_model)

    anthropic_base = os.environ.get("ANTHROPIC_BASE_URL", "").rstrip("/")
    anthropic_model = os.environ.get("ANTHROPIC_MODEL", "")
    if "openrouter.ai" in anthropic_base:
        token = (os.environ.get("OPENROUTER_API_KEY")
                 or os.environ.get("ANTHROPIC_AUTH_TOKEN", ""))
        if token and anthropic_model:
            return ("openai", anthropic_base + "/v1/chat/completions", token,
                    anthropic_model)
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key and anthropic_model:
        base = anthropic_base or "https://api.anthropic.com"
        return ("anthropic", base + "/v1/messages", key, anthropic_model)

    raise RuntimeError(
        "grounding found no model credentials: set OPENAI_BASE_URL/"
        "OPENAI_API_KEY/OPENAI_MODEL, or ANTHROPIC_API_KEY/ANTHROPIC_MODEL"
    )


def _grounding_request(provider, query, img_b64):
    kind, url, key, model = provider
    # max_tokens is bounded so a reasoning model cannot spend the whole
    # request thinking and return no coordinate.
    if kind == "anthropic":
        payload = json.dumps({
            "model": model, "max_tokens": 2000,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                                             "media_type": "image/png",
                                             "data": img_b64}},
                {"type": "text", "text": query},
            ]}],
        }).encode()
        headers = {"Content-Type": "application/json", "x-api-key": key,
                   "anthropic-version": "2023-06-01"}
        return url, payload, headers

    payload = json.dumps({
        "model": model, "max_tokens": 2000,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": query},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64," + img_b64}},
        ]}],
    }).encode()
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + key}
    return url, payload, headers


def _grounding_reply_text(provider, result):
    if provider[0] == "anthropic":
        return "".join(
            b.get("text", "") for b in result.get("content", [])
            if b.get("type") == "text"
        ).strip()
    return (result["choices"][0]["message"].get("content") or "").strip()


def call_grounding_model(screenshot_path, target_description, max_retries=3):
    """Locate a UI element on a screenshot and return its (x, y) coordinates.

    Grounding runs on **the same model this island's agent runs on**, picked up
    from the environment: OPENAI_* for an OpenAI-compatible deployment,
    otherwise ANTHROPIC_*. So a Qwen island grounds with Qwen and a Claude
    island grounds with Claude, and no island quietly borrows another model's
    vision. Returns the on-screen pixel coordinates of the described element
    as a string of the form ``"(x,y)"`` (no spaces, integer values).

    Accuracy differs sharply by model and is worth knowing before relying on
    it. Measured on a 1920x1080 desktop screenshot, 6 targets x 3 repeats:
    Claude Sonnet 5 hit 78% with a 7px median error, Qwen3.8 hit 50% with a
    54px median and outliers past 400px. Treat a returned coordinate as a
    hypothesis: click, re-screenshot, and check, rather than assuming it
    landed.

    The model is asked for coordinates normalised to [0,1] and this function
    scales them to pixels, so the returned value is always in real screen
    coordinates regardless of how the model resized the image internally.

    Example:
        >>> result = call_grounding_model("/tmp/screen.png", "the OK button")
        >>> result
        {'text': '(412,308)'}
        >>> import re
        >>> m = re.match(r"\\((\\d+),(\\d+)\\)", result['text'])
        >>> x, y = int(m.group(1)), int(m.group(2))

    Args:
        screenshot_path: path to a PNG screenshot of the current screen.
        target_description: a short noun-phrase describing the UI element to
            locate (e.g. "the blue OK button", "the search box at the top").
            Wrapped internally in the model's canonical query format —
            callers should NOT pre-format the string themselves.
        max_retries: retries on transient API errors / 429 rate limits.

    Returns:
        ``{"text": "(x,y)"}`` on success.
    """
    import re, struct, time, random
    provider = _grounding_provider()

    with open(screenshot_path, "rb") as f:
        raw = f.read()
    # PNG IHDR carries width/height at a fixed offset. Parsed from the header
    # rather than with Pillow so this helper keeps its stdlib-only dependency.
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {screenshot_path}")
    width, height = struct.unpack(">II", raw[16:24])
    img_b64 = base64.b64encode(raw).decode()

    # Ask for NORMALISED coordinates and scale them here. Asking this model for
    # raw pixels returns points in whatever space it resized the image to --
    # measured against a 1920x1080 desktop screenshot, every x came back at
    # ~0.52 of its true value and accuracy was 0/6. The same targets with
    # normalised output scored 4/5.
    query = (
        f"Find: {target_description}\n"
        f"Reply with only the coordinate as (x,y) where x and y are floats in "
        f"[0,1] relative to the image width and height."
    )
    url, payload, headers = _grounding_request(provider, query, img_b64)

    last_text = ""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers)
            resp = urllib.request.urlopen(req, timeout=120)
            last_text = _grounding_reply_text(provider, json.loads(resp.read()))
            m = re.search(r"\(?\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)?", last_text)
            if not m:
                if attempt < max_retries - 1:
                    continue
                raise ValueError(
                    f"no coordinate in grounding reply: {last_text[:200]!r}")
            fx, fy = float(m.group(1)), float(m.group(2))
            # Tolerate a model that ignored the instruction and gave pixels.
            x = int(round(fx * width)) if fx <= 1.0 else int(round(fx))
            y = int(round(fy * height)) if fy <= 1.0 else int(round(fy))
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            return {"text": f"({x},{y})"}
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                time.sleep(4.0 * (2 ** attempt) + random.uniform(0, 2))
                continue
            raise
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1.0)
                continue
            raise

def click_at(x, y):
    """Move the mouse pointer to (x, y) and left-click once (xdotool)."""
    subprocess.run(["xdotool", "mousemove", str(x), str(y), "click", "1"])

def key_press(keys):
    """Send a keypress sequence to the focused window (xdotool key syntax,
    e.g. "Return", "ctrl+s", "alt+F4")."""
    subprocess.run(["xdotool", "key", "--clearmodifiers", keys])

def get_active_window():
    """Return the title of the currently focused window as a string."""
    r = subprocess.run(["xdotool", "getactivewindow", "getwindowname"],
                       capture_output=True, text=True)
    return r.stdout.strip()
