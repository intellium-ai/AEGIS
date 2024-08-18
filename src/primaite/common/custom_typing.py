from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union

import yaml


class Serializable(ABC):

    @abstractmethod
    def serialize(self) -> list[dict] | dict[str, Any]: ...

    def save(self, path: str | Path):
        with open(path, "w+") as f:
            yaml.safe_dump(self.serialize(), f)
