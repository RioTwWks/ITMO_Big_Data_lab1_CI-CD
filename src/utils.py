import pandas as pd
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

def read_exp_data(config: ExperimentConfig):
    train_x = pd.read_csv(config.train_x_csv_path)
    train_y = pd.read_csv(config.train_y_csv_path).squeeze()
    test_x = pd.read_csv(config.test_x_csv_path)
    test_y = pd.read_csv(config.test_y_csv_path).squeeze()
    return train_x, train_y, test_x, test_y