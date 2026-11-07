"""Where the raw data lives.

Outside version control: Epinions alone is 386 MB, and third-party data is
downloaded from its source rather than redistributed. The location comes from
the DPCFOG_DATA environment variable and defaults to `data/` at the repository
root.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT = Path(__file__).resolve().parent.parent / "data"


def data_root() -> Path:
    return Path(os.environ.get("DPCFOG_DATA", DEFAULT))


def rev2_path(network: str) -> Path:
    return data_root() / "rev2" / "rev2-data" / network


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Run `python -m realdata.download` to fetch the networks, or point "
            "DPCFOG_DATA to where they already are."
        )
    return path
