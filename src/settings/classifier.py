from typing import Any, List, Optional, Sequence

from src.settings.base import ExtraFieldsNotAllowedBaseModel


class ClassifierSettings(ExtraFieldsNotAllowedBaseModel):
    n_estimators: int
    max_depth: Optional[int]


class FitInput(ExtraFieldsNotAllowedBaseModel):
    X: Sequence[str]
    y: Sequence[Any]


class PredictInput(ExtraFieldsNotAllowedBaseModel):
    X: Sequence[str]


class PredictOutput(ExtraFieldsNotAllowedBaseModel):
    predictions: Sequence[Any]
