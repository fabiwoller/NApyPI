from ._core import add

def add_numbers(a: int, b: int) -> int:
    """User-friendly wrapper around C++ add"""
    return add(a, b)
