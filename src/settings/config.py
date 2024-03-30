from typing import Optional

from src.settings.base import ExtraFieldsNotAllowedBaseModel
from src.settings.classifier import ClassifierSettings
from src.settings.preprocessor import PreprocessorSettings


class Config(ExtraFieldsNotAllowedBaseModel):
    preprocessing_config: PreprocessorSettings
    classifier_config: ClassifierSettings
