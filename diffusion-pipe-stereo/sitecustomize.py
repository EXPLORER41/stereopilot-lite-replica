"""Prefer this repository's top-level packages over similarly named deps."""

import importlib.util
import sys
from pathlib import Path


def _bind_local_package(name):
    root = Path(__file__).resolve().parent
    package_dir = root / name
    init_file = package_dir / "__init__.py"
    if not init_file.exists():
        return

    existing = sys.modules.get(name)
    existing_file = getattr(existing, "__file__", None)
    if existing_file:
        try:
            if Path(existing_file).resolve().is_relative_to(package_dir):
                return
        except OSError:
            pass

    spec = importlib.util.spec_from_file_location(
        name,
        init_file,
        submodule_search_locations=[str(package_dir)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


_bind_local_package("utils")

