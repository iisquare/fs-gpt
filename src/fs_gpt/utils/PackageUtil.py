import importlib.metadata
import importlib.util
from typing import TYPE_CHECKING

from packaging import version

if TYPE_CHECKING:
    from packaging.version import Version


class PackageUtil:
    @staticmethod
    def available(name: str) -> bool:
        return importlib.util.find_spec(name) is not None

    @staticmethod
    def version(name: str) -> Version:
        return version.parse(importlib.metadata.version(name))
