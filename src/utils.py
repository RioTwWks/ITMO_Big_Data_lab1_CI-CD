import yaml
from pydantic import ValidationError
from src.settings.config import ExperimentConfig


def load_experiment_config(config_path: str) -> ExperimentConfig:
    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)
    try:
        config = ExperimentConfig(**config_data)
        return config
    except ValidationError as e:
        print(f"Ошибка валидации конфигурации: {e}")
        raise
