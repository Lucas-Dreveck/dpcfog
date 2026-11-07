"""Download the real networks and the fraud labels.

    python -m realdata.download

Idempotent: files already in place are not downloaded again. At the end it
checks the counts against Table 2 of Kumar et al. (2018) and fails if they
differ.

Destination: `data/` at the repository root, or DPCFOG_DATA. About 250 MB
compressed, 800 MB uncompressed.

Sources, all public:

    SNAP, Stanford      Bitcoin OTC and Bitcoin Alpha signed networks
                        https://snap.stanford.edu/data/
    REV2 page           formatted networks and label files
                        https://cs.stanford.edu/~srijan/rev2/

The REV2 package holds OTC, Alpha, Epinions and Amazon with their `*_gt.csv`
label files, which are the labels used here. The SNAP files are a cross-check.
"""

from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

from .paths import data_root

SNAP_SOURCES = {
    "soc-sign-bitcoinotc.csv.gz": "https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz",
    "soc-sign-bitcoinalpha.csv.gz": "https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz",
}
REV2_SOURCES = {
    "rev2data.zip": "https://cs.stanford.edu/~srijan/rev2/rev2data.zip",
    "rev2codes.zip": "https://cs.stanford.edu/~srijan/rev2/rev2codes.zip",
}


def _download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        print(f"  already present: {destination.name}")
        return
    print(f"  downloading {destination.name} ...", flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)
    print(f"  ok: {destination.stat().st_size / 1e6:.1f} MB")


def _extract(zip_path: Path, into: Path) -> None:
    """Extract only the networks and the labels. The package's `.pkl` files are
    Python 2 pickles; the `.csv` files carry the same information."""
    with zipfile.ZipFile(zip_path) as z:
        members = [
            n for n in z.namelist()
            if n.endswith(("_network.csv", "_gt.csv")) or "rev2-code" in n
        ]
        for name in members:
            if (into / name).exists():
                continue
            z.extract(name, into)
    print(f"  extracted: {zip_path.name}")


def main() -> int:
    root = data_root()
    raw = root / "raw"
    print(f"destination: {root}")

    print("\nSNAP (cross-check):")
    for name, url in SNAP_SOURCES.items():
        _download(url, raw / name)

    print("\nREV2 (formatted networks and fraud labels):")
    for name, url in REV2_SOURCES.items():
        _download(url, raw / name)
    for name in REV2_SOURCES:
        _extract(raw / name, root / "rev2")

    print("\nchecking against Table 2 of Kumar et al. (2018):")
    from .networks import EXPECTED, load

    failures = 0
    for network in EXPECTED:
        try:
            net = load(network)
            print(f"  ok  {net.summary()}")
        except Exception as error:
            print(f"  FAIL {network}: {error}")
            failures += 1
    for network in ("epinions", "amazon"):
        try:
            net = load(network, validate=False)
            print(f"  ..  {net.summary()}")
        except Exception as error:
            print(f"  ..  {network}: unavailable ({error})")

    if failures:
        print("\nSome network does not match. Do NOT run experiments on it.")
        return 1
    print("\nAll counts match. Next step:")
    print("  python -m realdata.real_experiment otc alpha")
    return 0


if __name__ == "__main__":
    sys.exit(main())
