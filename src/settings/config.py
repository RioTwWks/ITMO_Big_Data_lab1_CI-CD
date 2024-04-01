from src.settings.base import ExtraFieldsNotAllowedBaseModel
from src.settings.classifier import ClassifierSettings
from src.settings.preprocessor import PreprocessorSettings


class Config(ExtraFieldsNotAllowedBaseModel):
    preprocessing_config: PreprocessorSettings
    classifier_config: ClassifierSettings

class ExperimentConfig(Config):
    exp_name: str
    log_path: str
    model_save_path: str
    vectorizer_save_path: str
