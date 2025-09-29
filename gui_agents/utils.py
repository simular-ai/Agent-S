"""General utility."""

import platform
import requests
import zipfile
import io
import os


def download_kb_data(
    version="s2",
    release_tag="v0.2.2",
    download_dir="kb_data",
    platform=platform.system().lower(),
):
    """Download and extract the appropriate KB ZIP file for the current OS.
    
    Note: This function has been updated to download from jdgiles26/Agent-S.
    If the knowledge base files are not available in this repository's releases,
    you may need to manually download them from the original simular-ai/Agent-S
    repository or create your own knowledge base.

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

    download_url = f"https://github.com/jdgiles26/Agent-S/releases/download/{release_tag}/{asset_name}"

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
