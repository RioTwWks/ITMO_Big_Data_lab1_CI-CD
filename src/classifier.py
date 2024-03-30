from typing import Any, Sequence

from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from src.settings.classifier import ClassifierSettings


class Classifier:
    def __init__(self, model_settings: ClassifierSettings, vectorizer=None, model=None):
        self.vectorizer = vectorizer if vectorizer else TfidfVectorizer()
        self.model = (
            model
            if model
            else RandomForestClassifier(
                n_estimators=model_settings.n_estimators,
                max_depth=model_settings.max_depth,
            )
        )

    def fit(self, X: Sequence[str], y: Sequence[Any]):
        X_transformed = self.vectorizer.fit_transform(X)
        self.model.fit(X_transformed, y)

    def predict(self, X: Sequence[str]) -> Sequence[Any]:
        X_transformed = self.vectorizer.transform(X)
        predictions = self.model.predict(X_transformed)
        return predictions

    def save(self, model_path: str, vectorizer_path: str):
        dump(self.model, model_path)
        dump(self.vectorizer, vectorizer_path)

    @classmethod
    def load(cls, model_path: str, vectorizer_path: str, model_settings: ClassifierSettings):
        model = load(model_path)
        vectorizer = load(vectorizer_path)
        return cls(model_settings=model_settings, vectorizer=vectorizer, model=model)
