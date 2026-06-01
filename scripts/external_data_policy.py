#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


def allowed_hpc_prefixes() -> tuple[Path, ...]:
    return (Path("/gpfsnyu") / "scratch", Path("/scratch"))


def is_hpc_scratch_path(path: Path) -> bool:
    resolved = path.expanduser().resolve()
    path_text = resolved.as_posix().rstrip("/") + "/"
    return any(path_text.startswith(prefix.as_posix().rstrip("/") + "/") for prefix in allowed_hpc_prefixes())


def enforce_hpc_external_path(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if is_hpc_scratch_path(resolved):
        return resolved
    allowed = ", ".join(prefix.as_posix() for prefix in allowed_hpc_prefixes())
    raise SystemExit(
        f"{description} must be stored on remote HPC scratch ({allowed}), not in the local checkout: {resolved}"
    )
