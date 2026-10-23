"""Route only the two official task packages, matching original editable layout."""
import importlib.abc
import importlib.util
from pathlib import Path
import sys


class OfficialTaskFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        base=Path(sys.prefix).resolve(strict=True).parents[1]/'upstream/EmbodiChain/embodichain_tasks'
        locations={'embodichain_tasks':base/'embodichain_tasks','embodichain_tasks.configs':base/'configs'}
        folder=locations.get(fullname)
        if folder is None:return None
        source=folder/'__init__.py'
        if not source.is_file() or source.is_symlink():
            raise ImportError('Verified official task package missing from installed portable root')
        return importlib.util.spec_from_file_location(fullname,source,submodule_search_locations=[str(folder)])


def install():
    # A new interpreter executes this only at site startup. No official source
    # is altered, no package stub is made, and arbitrary imports are untouched.
    if not any(type(value) is OfficialTaskFinder for value in sys.meta_path):
        sys.meta_path.insert(0,OfficialTaskFinder())
