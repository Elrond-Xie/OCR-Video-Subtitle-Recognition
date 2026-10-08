"""Configure project-local cache directories for packages and model downloads."""

from __future__ import annotations

import os
import shlex
from pathlib import Path


def configure_download_locations() -> Path:
    """Apply DOWNLOAD_DIR from the environment manifest before loading models."""
    project_root = Path(__file__).resolve().parents[2]
    manifest = project_root / "environment" / "ocr-versions.env"
    configured_dir: str | None = None
    try:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key.strip() == "DOWNLOAD_DIR":
                parsed = shlex.split(value.strip(), comments=False)
                configured_dir = parsed[0] if parsed else ""
                break
    except OSError:
        configured_dir = None

    raw_dir = os.environ.get("DOWNLOAD_DIR") or configured_dir or "downloads"
    download_root = Path(raw_dir).expanduser()
    if not download_root.is_absolute():
        download_root = project_root / download_root
    download_root = download_root.resolve()

    cache_locations = {
        "PIP_CACHE_DIR": download_root / "pip",
        "XDG_CACHE_HOME": download_root / "xdg",
        "HF_HOME": download_root / "huggingface",
        "HF_HUB_CACHE": download_root / "huggingface" / "hub",
        "PADDLE_PDX_CACHE_HOME": download_root / "paddlex",
        "TORCH_HOME": download_root / "torch",
    }
    os.environ["DOWNLOAD_DIR"] = str(download_root)
    for name, path in cache_locations.items():
        os.environ.setdefault(name, str(path))
        Path(os.environ[name]).expanduser().mkdir(parents=True, exist_ok=True)
    return download_root
