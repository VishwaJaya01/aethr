"""Dataset validation and lightweight preprocessing helpers for Aethr."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Optional

import yaml

LOGGER = logging.getLogger("aethr.data_preprocess")


def configure_logging(verbose: bool = False) -> None:
    """Configure module logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def find_dataset_yaml(raw_data_dir: Path) -> Optional[Path]:
    """Find a Roboflow YOLO `data.yaml` under the raw data directory."""
    candidates = sorted(raw_data_dir.rglob("data.yaml"))
    if not candidates:
        return None

    if len(candidates) > 1:
        LOGGER.warning(
            "Found multiple data.yaml files. Using first candidate: %s", candidates[0]
        )
    return candidates[0]


def validate_yolo_layout(dataset_root: Path) -> list[str]:
    """Validate common YOLO folder conventions and return missing paths."""
    required = [
        dataset_root / "train" / "images",
        dataset_root / "train" / "labels",
        dataset_root / "valid" / "images",
        dataset_root / "valid" / "labels",
    ]
    missing = [str(path) for path in required if not path.exists()]
    return missing


def copy_data_yaml(dataset_yaml: Path, processed_dir: Path) -> Path:
    """Copy dataset YAML into `data/processed/` for reproducible training configs."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    destination = processed_dir / "data.yaml"
    shutil.copy2(dataset_yaml, destination)
    return destination


def load_yaml(path: Path) -> dict:
    """Load YAML safely for sanity checks."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def run(raw_data_dir: Path, processed_dir: Path, strict: bool) -> int:
    """Run preprocessing checks and copy data yaml.

    Returns non-zero if strict checks fail.
    """
    dataset_yaml = find_dataset_yaml(raw_data_dir)
    if dataset_yaml is None:
        LOGGER.error("No data.yaml found under %s", raw_data_dir)
        return 1

    LOGGER.info("Using dataset YAML: %s", dataset_yaml)
    dataset_config = load_yaml(dataset_yaml)
    class_count = len(dataset_config.get("names", []))
    LOGGER.info("Detected %d classes in dataset config", class_count)

    missing_paths = validate_yolo_layout(dataset_yaml.parent)
    if missing_paths:
        LOGGER.warning("Dataset layout is missing expected paths:")
        for path in missing_paths:
            LOGGER.warning("  - %s", path)
        if strict:
            LOGGER.error("Strict mode enabled. Exiting due to missing paths.")
            return 2

    copied_yaml = copy_data_yaml(dataset_yaml, processed_dir)
    LOGGER.info("Copied data.yaml to %s", copied_yaml)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Prepare Aethr YOLO dataset metadata")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--strict", action="store_true", help="Fail when expected paths are missing")
    parser.add_argument("--verbose", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    configure_logging(verbose=args.verbose)
    raise SystemExit(run(raw_data_dir=args.raw_dir, processed_dir=args.processed_dir, strict=args.strict))
