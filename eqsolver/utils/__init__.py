from .symbolic_helpers import inverse_operator, integrate_with_conditions, apply_initial_conditions
from .conditions import extract_initial_conditions, build_homogeneous_solution, get_base_point, apply_initial_conditions as apply_cond

# Nota: apply_initial_conditions se exporta desde symbolic_helpers, pero también se puede desde conditions.
# Para evitar conflictos, elegimos la versión de symbolic_helpers como la principal.
__all__ = [
    "inverse_operator",
    "integrate_with_conditions",
    "apply_initial_conditions",   # desde symbolic_helpers
    "extract_initial_conditions",
    "build_homogeneous_solution",
    "get_base_point",
]