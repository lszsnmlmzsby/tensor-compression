__all__ = ["build_dataset", "build_dataloaders"]


def __getattr__(name: str):
    """Load HDF5-backed dataset builders only when callers request them.

    Importing a light-weight submodule such as ``tensor_compression.data.normalization``
    must not import the complete dataset registry.  Stage-2 adapter training consumes
    cached ``.pt`` latents and therefore has no HDF5 dependency, while the Stage-1 data
    loader still receives the same public ``build_dataset``/``build_dataloaders`` API.
    """

    if name in __all__:
        from importlib import import_module

        builders = import_module(".builders", __name__)
        return getattr(builders, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
