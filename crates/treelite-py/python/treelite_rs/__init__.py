"""treelite_rs — PyO3 binding for the Rust rewrite of Treelite.

Wave 0 (Plan 08-01) ships an empty importable surface: this package imports the
compiled ``_treelite_rs`` extension module and re-exports its ``frontend``,
``gtil``, and ``sklearn`` submodules (added via ``add_submodule`` they are not
auto-registered in ``sys.modules``, so we surface them here for
``from treelite_rs import frontend`` ergonomics — D-01 layout).

The re-exports are guarded so that ``import treelite_rs`` succeeds even before a
given symbol exists; downstream plans (08-02 .. 08-05) fill in ``Model``,
``TreeliteError``, the loaders, ``predict*``, and the sklearn estimator loaders.
"""

from . import _treelite_rs  # the compiled abi3 cdylib (maturin module-name)

__version__ = "0.1.0"

__all__ = ["_treelite_rs"]

# Re-export the compiled submodules when present. Guarded so the package imports
# cleanly during Wave 0 (the submodules exist but are empty) and as later plans
# add Model / TreeliteError / loaders without churning this file.
for _name in ("frontend", "gtil", "sklearn"):
    _sub = getattr(_treelite_rs, _name, None)
    if _sub is not None:
        globals()[_name] = _sub
        __all__.append(_name)
del _name, _sub

for _name in ("Model", "TreeliteError"):
    _obj = getattr(_treelite_rs, _name, None)
    if _obj is not None:
        globals()[_name] = _obj
        __all__.append(_name)
del _name, _obj
