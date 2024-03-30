from typing import Optional

from src.settings.base import ExtraFieldsNotAllowedBaseModel
from src.settings.preprocessor import PreprocessorSettings


class Config(ExtraFieldsNotAllowedBaseModel):
    preprocessing_config: PreprocessorSettings
