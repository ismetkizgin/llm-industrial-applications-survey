"""Download datasets from the reference GitHub repository."""

from __future__ import annotations

from pathlib import Path

import requests
from tqdm import tqdm

from src.config import DATA_DIR, DATASETS, GITHUB_RAW


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    dest.write_bytes(response.content)


def download_datasets(data_dir: Path | None = None) -> Path:
    data_dir = data_dir or DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    files: set[str] = set()
    for spec in DATASETS:
        files.update(spec.original_files)
        files.add(f"{spec.key}_train_mini.csv")
        files.add(f"{spec.key}_test_mini.csv")
        files.add(f"{spec.key}_train.csv")
        files.add(f"{spec.key}_test.csv")

    for name in sorted(files):
        url = f"{GITHUB_RAW}/{name}"
        dest = data_dir / name
        tqdm.write(f"Downloading {name}...")
        try:
            _download_file(url, dest)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                tqdm.write(f"  skipped (not in repo): {name}")
            else:
                raise

    return data_dir
