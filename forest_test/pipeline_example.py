from core import register_detector, Vectorizer, RandomForestMixer, Dataset
import numpy as np


@register_detector("dummy")
class DummyDetector:
    def __init__(self):
        self.stopvec = np.array([1])
        self._vec = np.zeros(1)

    @property
    def vec(self):
        return self._vec

    def __call__(self, local):
        self._vec[0] = 1.0 if local.get("flag") else 0.0


if __name__ == "__main__":
    vec = Vectorizer(["dummy"])
    local = {"flag": True}
    x = vec(local)
    print("Vector:", x)
