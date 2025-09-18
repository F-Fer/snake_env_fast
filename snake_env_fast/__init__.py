# snake_env_fast/__init__.py
from ._fastenv import BatchedEnv  # compiled pybind11 module

__all__ = ["BatchedEnv"]