from ._core import add, beta

def add_numbers(a: int, b: int) -> int:
    """User-friendly wrapper around C++ add"""
    return add(a, b)

def get_beta(dofs : int, value : float, times : int) -> float:
    return beta(dofs, value, times)
