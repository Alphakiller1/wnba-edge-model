from __future__ import annotations

from pathlib import Path


def repository_root(module_file: str | Path, *, cwd: Path | None = None) -> Path:
    """Resolve the checkout root without confusing Python's own ``data`` directory for it."""
    package_root = Path(module_file).resolve().parents[2]
    working_root = (cwd or Path.cwd()).resolve()
    for candidate in (package_root, working_root):
        if (candidate / "pyproject.toml").is_file() and (candidate / "data").is_dir():
            return candidate
    # Commands are documented to run from the checkout. Returning cwd also
    # gives callers a useful missing-file error rooted where the command ran.
    return working_root
