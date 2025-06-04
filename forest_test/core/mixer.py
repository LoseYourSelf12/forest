import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForestMixer:
    def __init__(self, **params):
        self.model = RandomForestClassifier(**params)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
