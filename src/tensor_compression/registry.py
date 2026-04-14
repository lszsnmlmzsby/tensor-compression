from __future__ import annotations

from collections.abc import Callable


class Registry:
    def __init__(self, name: str) -> None:
        self.name = name
        self._store: dict[str, Callable] = {}

    def register(self, key: str) -> Callable:
        def decorator(obj: Callable) -> Callable:
            if key in self._store:
                raise KeyError(f"{key!r} is already registered in {self.name}.")
            self._store[key] = obj
            return obj

        return decorator

    def get(self, key: str) -> Callable:
        if key not in self._store:
            available = ", ".join(sorted(self._store)) or "<empty>"
            raise KeyError(
                f"{key!r} is not registered in {self.name}. Available: {available}."
            )
        return self._store[key]

    def keys(self) -> list[str]:
        return sorted(self._store.keys())

