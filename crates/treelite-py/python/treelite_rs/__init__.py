"""treelite_rs — PyO3 binding for the Rust rewrite of Treelite.

Wave 1 (Plan 08-02) wires the walking-skeleton capability slice: load an XGBoost
(JSON/UBJSON/legacy) or LightGBM model and predict over a numpy array, matching
upstream ``treelite`` within 1e-5. This package imports the compiled
``_treelite_rs`` extension module, surfaces the single ``TreeliteError`` exception
and the ``Model`` pyclass at the top level, and exposes the pure-Python
``frontend`` / ``gtil`` shims (which port the upstream API shape over the compiled
str/bytes/zero-copy entry points — D-01 layout).

The compiled ``sklearn`` submodule stays empty until 08-04; its re-export is
guarded so ``import treelite_rs`` keeps working as later plans fill it in.
"""

from . import _treelite_rs  # the compiled abi3 cdylib (maturin module-name)
from . import frontend  # pure-Python path→bytes loader shim

__version__ = "0.1.0"

# Top-level surface: the single exception (D-06) + the Model pyclass.
TreeliteError = _treelite_rs.TreeliteError
Model = _treelite_rs.Model

__all__ = ["_treelite_rs", "frontend", "Model", "TreeliteError"]

# The `gtil` predict shim lands in 08-02 Task 2 (needs the compiled predict_f32/
# _f64 entry points). Guarded so Task-1 `import treelite_rs` succeeds before it.
try:
    from . import gtil  # noqa: F401

    __all__.append("gtil")
except ImportError:
    pass

# Re-export the compiled `sklearn` submodule when present (empty until 08-04).
# Guarded so the package imports cleanly before that slice lands.
_sklearn = getattr(_treelite_rs, "sklearn", None)
if _sklearn is not None:
    globals()["sklearn"] = _sklearn
    __all__.append("sklearn")
del _sklearn
