#!/usr/bin/env python3
"""Capture the frozen golden artifact from the upstream treelite==4.7.0 wheel.

Run ONCE on the environment where the Rust harness will run, then commit the
resulting ``fixtures/golden.json``. CI never regenerates this file (D-06/D-07).

Provenance / reproduction:
    pip install treelite==4.7.0 xgboost==3.2.0 numpy
    python fixtures/capture_golden.py

The script:
  1. Prints ``help(treelite.gtil.predict)`` so the exact 4.7.0 predict-kind
     keyword is confirmed against the installed wheel (RESEARCH Open Question #1).
  2. Loads the hand-crafted ``fixtures/binary_logistic.model.json`` into the
     upstream Treelite GTIL.
  3. Predicts the committed input matrix with the DEFAULT predict kind (sigmoid
     postprocessor applied -> outputs in the open interval (0, 1), Assumption A2).
  4. Writes ``{input, output, manifest}`` to ``fixtures/golden.json``. The
     manifest records treelite/xgboost versions, OS, arch, and libc/glibc
     version so a future libm-divergence failure is diagnosable (D-07).
"""

import json
import os
import platform

import numpy as np
import treelite
import treelite.gtil
import xgboost

# Resolve paths relative to this script so it works from any cwd.
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "binary_logistic.model.json")
GOLDEN_PATH = os.path.join(HERE, "golden.json")


def main() -> None:
    # (1) Confirm the 4.7.0 predict-kind keyword (RESEARCH Open Question #1).
    print("=== help(treelite.gtil.predict) ===")
    help(treelite.gtil.predict)
    print("=== end help ===")

    # (2) Load the hand-crafted XGBoost-JSON fixture into upstream Treelite.
    #     If this raises, the fixture nesting is wrong (Pitfall 5) -- repair the
    #     JSON against delegated_handler.cc:484-490 and re-run.
    model = treelite.frontend.load_xgboost_model(MODEL_PATH)

    # (3) The committed input matrix: 2 features per row. Rows are chosen to
    #     exercise both split directions in both trees plus a NaN (missing)
    #     value that must route via default_left.
    #
    #     Tree 0 splits on feature[0] at 0.5 (default_left=1);
    #     Tree 1 splits on feature[1] at 1.5 (default_left=1).
    X = np.array(
        [
            [0.0, 0.0],          # f0<0.5 -> -0.75 ; f1<1.5 ->  0.50
            [1.0, 2.0],          # f0>=0.5-> 1.25  ; f1>=1.5-> -0.25
            [0.0, 2.0],          # f0<0.5 -> -0.75 ; f1>=1.5-> -0.25
            [1.0, 0.0],          # f0>=0.5-> 1.25  ; f1<1.5 ->  0.50
            [float("nan"), 0.0],  # f0 missing -> default_left -> -0.75 ; f1<1.5 -> 0.50
        ],
        dtype=np.float32,
    )

    # (4) Default predict kind => sigmoid postprocessor applied (Assumption A2).
    #     Per Open Question #1, 4.7.0 uses pred_margin=False for the default kind.
    y = treelite.gtil.predict(model, X, pred_margin=False)
    output = np.asarray(y).ravel().tolist()

    manifest = {
        "treelite": treelite.__version__,      # expect 4.7.0
        "xgboost": xgboost.__version__,        # expect 3.2.0
        "os": platform.platform(),
        "arch": platform.machine(),
        "libc": platform.libc_ver(),           # glibc version on Linux
        "python": platform.python_version(),
    }

    golden = {
        "input": X.tolist(),
        "output": output,
        "manifest": manifest,
    }

    with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2)
        f.write("\n")

    # Provenance / sanity echo. The harness's real assertion is the 1e-5 check;
    # this just confirms the sigmoid path fired (all outputs in (0, 1)).
    assert all(0.0 < v < 1.0 for v in output), (
        f"sigmoid output must be in (0,1), got {output}"
    )
    print(f"Wrote {GOLDEN_PATH}")
    print(f"input rows : {len(golden['input'])}")
    print(f"output     : {output}")
    print(f"manifest   : {manifest}")


if __name__ == "__main__":
    main()
