"""Asset resolution: bundled defaults with project-local overrides.

Every suite and rubric reference flows through `find_suite` / `find_rubric`.
Resolution order:

1. If the reference looks like a path (contains a separator or `.yaml`, or is
   absolute), treat it literally. Return the resolved path if it exists,
   otherwise raise FileNotFoundError.
2. Otherwise, look in `<cwd>/suites/<name>.yaml` (or `rubrics/`). This is the
   project-local override slot.
3. Otherwise, fall back to the packaged default shipped inside
   `feral.data.suites` / `feral.data.rubrics`.

This lets `pip install feral && feral run -s coding-basics` work from any
directory, while developers editing YAML in the repo see their changes
immediately (editable install exposes the live `src/feral/data/` tree).
"""

from __future__ import annotations

import atexit
import importlib.metadata
import importlib.resources
from contextlib import ExitStack
from pathlib import Path

_resource_stack = ExitStack()
atexit.register(_resource_stack.close)


def _packaged_dir(subpkg: str) -> Path:
    """Return a real filesystem Path for a packaged data subdirectory.

    Uses `as_file()` so zipimport installs also work; for normal on-disk
    wheels this is a zero-copy passthrough.
    """
    traversable = importlib.resources.files(f"feral.data.{subpkg}")
    return Path(_resource_stack.enter_context(importlib.resources.as_file(traversable)))


PACKAGED_SUITES_DIR: Path = _packaged_dir("suites")
PACKAGED_RUBRICS_DIR: Path = _packaged_dir("rubrics")


def is_pathlike(ref: str | Path) -> bool:
    """True if the reference should be treated as a literal filesystem path.

    Bare names like `coding-basics` are looked up by name; anything with a
    separator, `.yaml` extension, or absolute prefix is treated as a path.
    """
    s = str(ref)
    if isinstance(ref, Path):
        return True
    return (
        "/" in s
        or "\\" in s
        or s.endswith(".yaml")
        or s.endswith(".yml")
        or Path(s).is_absolute()
    )


def _find_by_name(
    name: str,
    cwd_dir: str,
    packaged_dir: Path,
) -> Path:
    """Look up `<name>.yaml` first in cwd_dir, then in packaged_dir."""
    cwd_candidate = Path.cwd() / cwd_dir / f"{name}.yaml"
    if cwd_candidate.exists():
        return cwd_candidate
    packaged_candidate = packaged_dir / f"{name}.yaml"
    if packaged_candidate.exists():
        return packaged_candidate
    raise FileNotFoundError(
        f"Could not find {cwd_dir[:-1]} '{name}'. Tried:\n"
        f"  {cwd_candidate}\n"
        f"  {packaged_candidate}"
    )


def _find_asset(
    name_or_path: str | Path,
    cwd_dir: str,
    packaged_dir: Path,
) -> Path:
    """Resolve an asset reference. See module docstring for order."""
    if is_pathlike(name_or_path):
        path = Path(name_or_path)
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(f"File not found: {path}")
    return _find_by_name(str(name_or_path), cwd_dir, packaged_dir)


def find_suite(name_or_path: str | Path) -> Path:
    """Resolve a suite reference to an existing YAML file.

    Accepts either a bare name (`coding-basics`) or an explicit path
    (`./my-suite.yaml`, `src/feral/data/suites/coding-basics.yaml`).
    """
    return _find_asset(name_or_path, "suites", PACKAGED_SUITES_DIR)


def find_rubric(name_or_path: str | Path) -> Path:
    """Resolve a rubric reference to an existing YAML file.

    Accepts either a bare name (`default`, `coding`) or an explicit path.
    """
    return _find_asset(name_or_path, "rubrics", PACKAGED_RUBRICS_DIR)


def resolve_suite_dir(override: Path | None = None) -> Path:
    """Pick the directory used for interactive suite pickers and auto-discovery.

    Resolution: explicit override > `<cwd>/suites` (if it exists) > packaged suites dir.
    """
    if override is not None:
        return override
    cwd_dir = Path.cwd() / "suites"
    if cwd_dir.is_dir():
        return cwd_dir
    return PACKAGED_SUITES_DIR


def resolve_rubric_dir(override: Path | None = None) -> Path:
    """Pick the directory used for category-based rubric loading.

    Resolution: explicit override > `<cwd>/rubrics` (if it exists) > packaged rubrics dir.
    """
    if override is not None:
        return override
    cwd_dir = Path.cwd() / "rubrics"
    if cwd_dir.is_dir():
        return cwd_dir
    return PACKAGED_RUBRICS_DIR


def feral_version() -> str:
    """Return the installed package version for reproducibility metadata."""
    try:
        return importlib.metadata.version("feral")
    except importlib.metadata.PackageNotFoundError:
        try:
            import feral

            return getattr(feral, "__version__", "unknown")
        except Exception:
            return "unknown"
