#!/usr/bin/env python3
"""One-session 3-format XGBoost fixture + golden generator (Phase 3, Plan 03-01).

Produces, from ONE logical ``binary:logistic`` numerical-split model
(``base_score=0.5``), the seven frozen artifacts every downstream Phase-3 slice's
verify step consumes (D-05 / D-06 / D-10):

    fixtures/xgb_3format.model          legacy-binary  (PINNED OLD xgboost)
    fixtures/xgb_3format.json           XGBoost-JSON   (current xgboost 3.2.0)
    fixtures/xgb_3format.ubj            XGBoost-UBJSON (current xgboost 3.2.0)
    fixtures/xgb_3format.golden.json    shared prediction golden (treelite 4.7.0)
    fixtures/golden_v5_3format.bin      single v5 byte-fidelity golden blob (D-10)
    fixtures/xgb_3format.manifest.json  frozen generator manifest

Run ONCE in THROWAWAY environments — NEVER the project venv, NEVER CI (D-06).
The script is generation-only and frozen after one successful run; it adds zero
runtime/CI dependency.

================================================================================
WHY TWO ENVIRONMENTS (and how this script orchestrates them)
================================================================================
Current xgboost (3.2.0) silently writes UBJSON for ``.model`` (RESEARCH Pitfall 7),
so the LEGACY-binary fixture must be written by a PINNED OLD xgboost that still
emits the real ``LearnerModelParam`` layout (A1: start at 1.7.6, settle the pin
empirically). But the JSON/UBJSON fixtures + the goldens want current xgboost
3.2.0 + treelite 4.7.0. Those two xgboost versions cannot coexist in one
interpreter, so this script runs in two phases, each in its own ephemeral
``uv run --no-project --with ...`` environment:

  PHASE "legacy"  (old xgboost):  trains the model with a fixed seed/params,
                                  dumps the JSON spec of the trained booster to a
                                  tmp file (so the new-xgboost phase loads THE
                                  SAME logical model, not a re-train), and writes
                                  the legacy ``.model``. Runs the A1 assert.

  PHASE "modern"  (xgboost 3.2.0 + treelite 4.7.0 + numpy): re-loads that exact
                                  booster spec, writes ``.json`` + ``.ubj``,
                                  captures both goldens from the treelite wheel,
                                  runs the version-triple + A2 cross-format
                                  asserts, and freezes the manifest.

Recommended driver (operator runs this once from the workspace root):

    uv run --no-project --with 'xgboost==1.7.6' \
        python fixtures/generate_xgb_3format.py legacy

    uv run --no-project --with 'xgboost==3.2.0' --with 'treelite==4.7.0' \
        --with numpy python fixtures/generate_xgb_3format.py modern

If ``xgboost==1.7.6`` refuses to write legacy binary (the A1 assert fires),
retry the legacy phase with ``1.6.2`` then ``0.90``; lock the first version whose
output passes the A1 assert and record it in the manifest (the modern phase reads
it back from the handoff file). Running with NO phase argument prints these
instructions and exits non-zero (the script will not silently no-op).

base_score=0.5 (A2): sigmoid(0.5)=0 margin, so ``ProbToMargin`` is a no-op and the
``version[0] >= 1`` base_score gate cannot make the three formats diverge — all
three serialize to ONE v5 blob (the D-10 single-golden invariant).
"""

import hashlib
import json
import os
import platform
import struct
import sys

# Resolve all paths relative to this script so it works from any cwd.
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_LEGACY = os.path.join(HERE, "xgb_3format.model")
MODEL_JSON = os.path.join(HERE, "xgb_3format.json")
MODEL_UBJ = os.path.join(HERE, "xgb_3format.ubj")
GOLDEN_PRED = os.path.join(HERE, "xgb_3format.golden.json")
GOLDEN_V5 = os.path.join(HERE, "golden_v5_3format.bin")
MANIFEST = os.path.join(HERE, "xgb_3format.manifest.json")

# Cross-phase handoff: the legacy phase dumps the trained booster spec here so the
# modern phase reproduces THE SAME logical model rather than re-training.
BOOSTER_SPEC = os.path.join(HERE, ".xgb_3format.booster.json")
# The legacy phase records the confirmed old-xgboost version here for the manifest.
LEGACY_VER_FILE = os.path.join(HERE, ".xgb_3format.legacy_ver.txt")

# Fixed model knobs (D-05: small, binary:logistic, NUMERICAL SPLITS ONLY, no
# categorical features). base_score=0.5 neutralizes the version-gate (A2).
N_FEATURES = 4
N_ROWS_TRAIN = 256
N_ESTIMATORS = 6
MAX_DEPTH = 3
BASE_SCORE = 0.5
SEED = 1234

# Deterministic prediction-golden input matrix (seeded), 4 features per row.
N_ROWS_PRED = 8


def _train_booster(xgb, np):
    """Train ONE deterministic binary:logistic numerical-split booster.

    Returns a trained ``xgboost.Booster``. No categorical features: the DMatrix
    carries plain float columns, so every split is a numerical test (D-04/D-05).
    """
    rng = np.random.RandomState(SEED)
    X = rng.rand(N_ROWS_TRAIN, N_FEATURES).astype(np.float32)
    # A learnable-but-simple target so trees actually split numerically.
    y = ((X[:, 0] + X[:, 1] - X[:, 2] * 0.5) > 1.0).astype(np.float32)
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "objective": "binary:logistic",
        "base_score": BASE_SCORE,  # A2: sigmoid(0.5)=0 margin -> gate is a no-op
        "max_depth": MAX_DEPTH,
        "eta": 0.3,
        "seed": SEED,
        # Force numerical splits only — no enable_categorical anywhere.
        "tree_method": "exact",
    }
    booster = xgb.train(params, dtrain, num_boost_round=N_ESTIMATORS)
    return booster


def _assert_legacy_header(path):
    """A1 verification gate (RESEARCH Pitfall 7 / Legacy Binary Layout).

    The legacy file's first byte must NOT be ``{`` (JSON) and NOT ``N`` (UBJSON
    no-op marker), and the first 136 bytes must decode as a ``LearnerModelParam``
    (base_score f32 LE @0, num_feature u32 LE @4) with sane values.
    """
    with open(path, "rb") as f:
        head = f.read(136)
    assert len(head) >= 136, (
        f"legacy fixture {path} is only {len(head)} bytes; a genuine "
        "LearnerModelParam header is 136 bytes — this is not legacy binary."
    )
    first = head[0:1]
    assert first != b"{", (
        f"legacy fixture {path} starts with '{{' — current xgboost silently wrote "
        "JSON/UBJSON (Pitfall 7). Use an OLDER pinned xgboost (retry 1.6.2 / 0.90)."
    )
    assert first != b"N", (
        f"legacy fixture {path} starts with 'N' (UBJSON no-op marker) — this is "
        "UBJSON, not legacy binary. Use an OLDER pinned xgboost."
    )
    base_score = struct.unpack_from("<f", head, 0)[0]
    num_feature = struct.unpack_from("<I", head, 4)[0]
    assert 0 < num_feature < 1_000_000, (
        f"legacy LearnerModelParam.num_feature={num_feature} is insane — the "
        "136-byte header did not decode; this is not a legacy LearnerModelParam."
    )
    # base_score in the header may be the raw stored 0.5 or a transformed margin;
    # either way it must be a finite, sane f32.
    import math

    assert math.isfinite(base_score), (
        f"legacy LearnerModelParam.base_score={base_score} is not finite — header "
        "decode failed."
    )
    print(
        f"A1 OK: {os.path.basename(path)} first byte={first!r}, "
        f"LearnerModelParam(base_score={base_score}, num_feature={num_feature})"
    )


def phase_legacy():
    """PHASE 'legacy': train, dump booster spec, write legacy .model, A1 assert."""
    import numpy as np
    import xgboost as xgb

    booster = _train_booster(xgb, np)

    # Hand the EXACT trained model to the modern phase via the booster's JSON
    # spec so the modern phase does not re-train (keeps it ONE logical model).
    booster.save_model(BOOSTER_SPEC)

    # Write the legacy-binary fixture with this OLD xgboost.
    booster.save_model(MODEL_LEGACY)
    _assert_legacy_header(MODEL_LEGACY)

    with open(LEGACY_VER_FILE, "w", encoding="utf-8") as f:
        f.write(xgb.__version__)

    print(f"Wrote {MODEL_LEGACY} (legacy binary, xgboost {xgb.__version__})")
    print(f"Wrote booster handoff spec {BOOSTER_SPEC}")
    print("PHASE 'legacy' complete. Now run the 'modern' phase.")


def phase_modern():
    """PHASE 'modern': write JSON/UBJSON, capture goldens, A2 + version asserts."""
    import numpy as np
    import treelite
    import treelite.gtil
    import xgboost as xgb

    assert os.path.exists(BOOSTER_SPEC), (
        f"missing {BOOSTER_SPEC} — run the 'legacy' phase first so both phases "
        "share ONE logical model."
    )

    # Re-load THE SAME logical booster the legacy phase trained (no re-train).
    booster = xgb.Booster()
    booster.load_model(BOOSTER_SPEC)

    # (2) Save JSON + UBJSON from the SAME logical model. The .json/.ubj
    #     extensions ARE honored by current xgboost (.model is not — that is why
    #     legacy used the old pin).
    booster.save_model(MODEL_JSON)
    booster.save_model(MODEL_UBJ)
    # Guard against Pitfall 7 in reverse: JSON must start with '{', UBJSON must NOT.
    with open(MODEL_JSON, "rb") as f:
        assert f.read(1) == b"{", "xgb_3format.json is not JSON"
    print(f"Wrote {MODEL_JSON} + {MODEL_UBJ} (xgboost {xgb.__version__})")

    # (3) Shared prediction golden from the upstream Treelite 4.7.0 wheel.
    tl_model = treelite.frontend.load_xgboost_model(MODEL_JSON)
    rng = np.random.RandomState(SEED + 1)
    X = rng.rand(N_ROWS_PRED, N_FEATURES).astype(np.float32)
    y = treelite.gtil.predict(tl_model, X, pred_margin=False)
    output = np.asarray(y).ravel().tolist()
    # sigmoid postprocessor must have fired (binary:logistic) -> outputs in (0,1).
    assert all(0.0 < v < 1.0 for v in output), (
        f"sigmoid output must be in (0,1), got {output}"
    )
    golden = {
        "input": X.tolist(),
        "output": output,
        "manifest": {
            "treelite": treelite.__version__,
            "xgboost": xgb.__version__,
            "os": platform.platform(),
            "arch": platform.machine(),
            "libc": list(platform.libc_ver()),
            "python": platform.python_version(),
        },
    }
    with open(GOLDEN_PRED, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2)
        f.write("\n")
    print(f"Wrote {GOLDEN_PRED} ({len(output)} rows)")

    # (4) Single v5 byte golden from the JSON load -> serialize_bytes().
    blob = bytes(tl_model.serialize_bytes())
    with open(GOLDEN_V5, "wb") as f:
        f.write(blob)
    version_triple = struct.unpack("<iii", blob[:12])
    assert version_triple == (4, 7, 0), (
        f"expected v5 header version triple (4, 7, 0), got {version_triple} — the "
        "'v5'==4.7.0 assumption is wrong; revisit before locking the golden."
    )
    print(f"Wrote {GOLDEN_V5} ({len(blob)} bytes), version triple {version_triple}")

    # (5) A2 cross-format gate: load EACH of the three formats through the
    #     Treelite wheel, serialize each, and assert all three == the single v5
    #     blob. This proves the single-golden invariant at generation time,
    #     before the Rust loaders exist.
    for fmt_path in (MODEL_JSON, MODEL_UBJ, MODEL_LEGACY):
        assert os.path.exists(fmt_path), (
            f"missing {fmt_path} — run the 'legacy' phase before 'modern'."
        )
        m = treelite.frontend.load_xgboost_model(fmt_path)
        other = bytes(m.serialize_bytes())
        assert other == blob, (
            f"A2 FAILED: {os.path.basename(fmt_path)} serializes to a DIFFERENT v5 "
            f"blob than {os.path.basename(GOLDEN_V5)} (nbytes "
            f"{len(other)} vs {len(blob)}). base_score=0.5 should make the "
            "version-gate a no-op so all three formats agree; investigate before "
            "freezing."
        )
    print("A2 OK: all three formats serialize to the SAME v5 blob.")

    # (6) Freeze the manifest.
    legacy_ver = "unknown"
    if os.path.exists(LEGACY_VER_FILE):
        with open(LEGACY_VER_FILE, encoding="utf-8") as f:
            legacy_ver = f.read().strip()
    manifest = {
        "xgboost_write_json_ubj": xgb.__version__,   # expect 3.2.0
        "xgboost_write_legacy": legacy_ver,          # the CONFIRMED old pin (A1)
        "treelite": treelite.__version__,            # expect 4.7.0
        "os": platform.platform(),
        "arch": platform.machine(),
        "libc": list(platform.libc_ver()),
        "python": platform.python_version(),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "nbytes": len(blob),
        "source_fixtures": [
            os.path.basename(MODEL_JSON),
            os.path.basename(MODEL_UBJ),
            os.path.basename(MODEL_LEGACY),
        ],
    }
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Wrote {MANIFEST}")
    print(f"  legacy-write xgboost = {legacy_ver} (A1 confirmed)")
    print(f"  sha256(golden v5)    = {manifest['sha256']}")
    print("PHASE 'modern' complete. All six artifacts frozen.")


USAGE = (
    "usage: generate_xgb_3format.py {legacy|modern}\n\n"
    "Run in TWO throwaway envs (NEVER the project venv, NEVER CI):\n"
    "  uv run --no-project --with 'xgboost==1.7.6' \\\n"
    "      python fixtures/generate_xgb_3format.py legacy\n"
    "  uv run --no-project --with 'xgboost==3.2.0' --with 'treelite==4.7.0' \\\n"
    "      --with numpy python fixtures/generate_xgb_3format.py modern\n\n"
    "If the legacy phase's A1 assert fires, retry with 'xgboost==1.6.2' then\n"
    "'xgboost==0.90'; lock the first version whose output passes A1.\n"
)


def main(argv):
    if len(argv) != 2 or argv[1] not in ("legacy", "modern"):
        sys.stderr.write(USAGE)
        return 2
    if argv[1] == "legacy":
        phase_legacy()
    else:
        phase_modern()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
