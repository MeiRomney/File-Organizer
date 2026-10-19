"""Common interface so all approaches are trained and tested the same way."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from fileorg.models import FileRecord


class Classifier(ABC):
    name: str = "classifier"

    def fit(self, records: Sequence[FileRecord]) -> None:
        """Train on labelled records. Rule-based classifiers can skip this."""

    @abstractmethod
    def predict(self, record: FileRecord) -> str:
        """Return a category name for one record."""

    def predict_many(self, records: Sequence[FileRecord]) -> list[str]:
        return [self.predict(r) for r in records]
