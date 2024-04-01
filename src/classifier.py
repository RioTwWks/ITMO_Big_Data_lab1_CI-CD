import os
from os.path import join as opj
from typing import Any, Sequence, Union

from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from src.settings.classifier import ClassifierSettings, PredictOutput


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
        self.id2label = model_settings.id2label

    def fit(self, X: Sequence[str], y: Sequence[Any]):
        X_transformed = self.vectorizer.fit_transform(X)
        self.model.fit(X_transformed, y)

    def predict(self, X: Union[str, Sequence[str]]) -> PredictOutput:
        if isinstance(X, str):
            X = [X]
        X_transformed = self.vectorizer.transform(X)
        numeric_predictions = self.model.predict(X_transformed)

        text_predictions = [self.id2label[id] for id in numeric_predictions]
        if len(text_predictions) == 1:
            return PredictOutput(sentiment=text_predictions[0])
        return PredictOutput(sentiment=text_predictions)

    def save(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        model_path = opj(save_path, "model.joblib")
        vectorizer_path = opj(save_path, "vectorizer.joblib")
        dump(self.model, model_path)
        dump(self.vectorizer, vectorizer_path)

    @classmethod
    def load(cls, load_path: str, model_settings: ClassifierSettings):
        model_path = opj(load_path, "model.joblib")
        vectorizer_path = opj(load_path, "vectorizer.joblib")
        model = load(model_path)
        vectorizer = load(vectorizer_path)
        return cls(model_settings=model_settings, vectorizer=vectorizer, model=model)
