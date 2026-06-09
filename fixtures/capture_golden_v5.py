#!/usr/bin/env python3
"""Capture the frozen D-02 golden v5 serialized blob from the upstream wheel.

Run ONCE on the environment where the Rust serializer round-trip is validated,
then commit the resulting ``fixtures/golden_v5.bin`` + ``fixtures/golden_v5.manifest.json``.
CI never regenerates these files (mirrors the Phase 1 ``capture_golden.py``
"run once, commit, CI never regenerates" discipline).

Provenance / reproduction (the repo .venv already has these installed):
    uv run python fixtures/capture_golden_v5.py
    # or, with a plain venv on PATH:
    #   pip install treelite==4.7.0 xgboost==3.2.0 numpy
    #   python fixtures/capture_golden_v5.py

The script:
  1. Loads the SAME hand-crafted ``fixtures/binary_logistic.model.json`` used by
     Phase 1 (keeps the sigmoid/binary:logistic path exercised) into upstream
     Treelite via ``treelite.frontend.load_xgboost_model`` (the exact call the
     Phase 1 capture used successfully).
  2. Calls ``model.serialize_bytes()`` to obtain the authoritative v5 byte stream
     (routes to ``TreeliteSerializeModelToBytes`` -> ``SerializeToBuffer``).
  3. Writes the raw bytes verbatim to ``fixtures/golden_v5.bin`` (binary mode).
  4. Writes ``fixtures/golden_v5.manifest.json`` with the Phase 1 manifest keys
     (treelite/xgboost/os/arch/libc/python) PLUS ``sha256``, ``nbytes``, and
     ``source_fixture`` so a future byte-divergence failure is diagnosable.

The blob's first 12 bytes are the version triple as little-endian int32. They
MUST decode to ``(4, 7, 0)`` (RESEARCH Summary finding 1 / Pitfall 1): "v5" names
the wire generation, but the 4.7.0 wheel stamps ``major_ver=4``. If they decode
to ``(5, x, x)`` the version-constant assumption (A1) is wrong and the serializer
constants must be revisited before any further serialization work. The script
asserts this so the capture itself settles the assumption empirically.
"""

import hashlib
import json
import os
import platform
import struct

import treelite
import xgboost

# Resolve paths relative to this script so it works from any cwd.
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "binary_logistic.model.json")
BLOB_PATH = os.path.join(HERE, "golden_v5.bin")
MANIFEST_PATH = os.path.join(HERE, "golden_v5.manifest.json")


def main() -> None:
    # (1) Load the hand-crafted XGBoost-JSON fixture into upstream Treelite,
    #     exactly as the Phase 1 capture did.
    model = treelite.frontend.load_xgboost_model(MODEL_PATH)

    # (2) The authoritative v5 byte stream (do NOT hand-fabricate — it must come
    #     from the wheel).
    blob = model.serialize_bytes()
    blob = bytes(blob)

    # (3) Freeze the raw bytes.
    with open(BLOB_PATH, "wb") as f:
        f.write(blob)

    # (4) Manifest: Phase 1 keys + sha256/nbytes/source_fixture.
    manifest = {
        "treelite": treelite.__version__,   # expect 4.7.0
        "xgboost": xgboost.__version__,     # expect 3.2.0
        "os": platform.platform(),
        "arch": platform.machine(),
        "libc": list(platform.libc_ver()),  # glibc version on Linux
        "python": platform.python_version(),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "nbytes": len(blob),
        "source_fixture": "binary_logistic.model.json",
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    # Empirically settle the version-header assumption (A1): first 12 bytes are
    # the version triple as little-endian int32. MUST be (4, 7, 0).
    version_triple = struct.unpack("<iii", blob[:12])
    assert version_triple == (4, 7, 0), (
        f"expected v5 header version triple (4, 7, 0), got {version_triple} -- "
        "the 'v5'==4.7.0 assumption (A1) is wrong; revisit serializer constants "
        "before any further serialization work."
    )

    print(f"Wrote {BLOB_PATH} ({len(blob)} bytes)")
    print(f"Wrote {MANIFEST_PATH}")
    print(f"sha256         : {manifest['sha256']}")
    print(f"version triple : {version_triple}")
    print(f"manifest       : {manifest}")


if __name__ == "__main__":
    main()
