"""Type stub for the treelite_rs.Model pyclass (PEP 561, D-10)."""

from __future__ import annotations

class Model:
    """A loaded tree-ensemble model (owns the Rust ``treelite_core::Model``)."""

    @property
    def num_tree(self) -> int: ...
    @property
    def num_feature(self) -> int: ...
    @property
    def input_type(self) -> str: ...
    @property
    def output_type(self) -> str: ...
