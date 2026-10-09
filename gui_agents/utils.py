"""General utility."""

import platform
import re
import requests
import zipfile
import io
import os

# Anthropic models that reject non-default sampling params (temperature, top_p, top_k).
_ANTHROPIC_NO_TEMPERATURE_PATTERNS = (
    re.compile(r"sonnet-5"),
    re.compile(r"opus-4-7"),
    re.compile(r"opus-4-8"),
)


def anthropic_supports_temperature(model: str) -> bool:
    """Return False for Anthropic models that reject the temperature parameter."""
    model_lower = model.lower()
    return not any(
        pattern.search(model_lower) for pattern in _ANTHROPIC_NO_TEMPERATURE_PATTERNS
    )


def extract_anthropic_text(response) -> str:
    """Concatenate text from an Anthropic Messages response.

    Newer models (e.g. Sonnet 5) enable adaptive thinking by default, so the
    response content can contain ThinkingBlocks that have no `text` attribute.
    This skips non-text blocks and returns only the assistant's text output.
    """
    texts = [
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    ]
    return "".join(texts)


def extract_anthropic_thinking(response) -> str:
    """Concatenate thinking/reasoning text from an Anthropic Messages response."""
    thoughts = [
        getattr(block, "thinking", "")
        for block in response.content
        if getattr(block, "type", None) == "thinking"
    ]
    return "".join(thoughts)


def download_kb_data(
    version="s2",
    release_tag="v0.2.2",
    download_dir="kb_data",
    platform=platform.system().lower(),
):
    """Download and extract the appropriate KB ZIP file for the current OS.

    Args:
        version (str): Prefix in the asset name (e.g., "s1" or "s2")
        release_tag (str): Tag of the release that has the assets (e.g., "v0.2.2")
        download_dir (str): Where to extract the downloaded files
        platform (str): OS (e.g., "windows", "darwin", "linux")
    """
    # Detect OS
    if platform not in ["windows", "darwin", "linux"]:
        raise RuntimeError(f"Unsupported OS: {platform}")

    # Build asset filename, e.g. "s1_windows.zip" or "s1_darwin.zip"
    asset_name = f"{version}_{platform}.zip"

    download_url = f"https://github.com/simular-ai/Agent-S/releases/download/{release_tag}/{asset_name}"

    # Make sure our output directory exists
    os.makedirs(download_dir, exist_ok=True)

    print(f"Downloading {asset_name} from {download_url} ...")
    response = requests.get(download_url)
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to download {asset_name}. "
            f"HTTP status: {response.status_code} - {response.reason}"
        )

    # Extract the ZIP in-memory
    zip_data = io.BytesIO(response.content)
    with zipfile.ZipFile(zip_data, "r") as zip_ref:
        zip_ref.extractall(download_dir)

    print(f"Extracted {asset_name} to ./{download_dir}")
