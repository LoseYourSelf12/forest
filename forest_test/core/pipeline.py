from typing import List, Dict, Any
import numpy as np
from .registry import DetectorRegistry


class Vectorizer:
    def __init__(self, detector_names: List[str]):
        self.detectors = [DetectorRegistry.create(name) for name in detector_names]

    def __call__(self, local: Dict[str, Any]) -> np.ndarray:
        vecs = []
        for det in self.detectors:
            det(local)
            vecs.append(det.vec)
        if vecs:
            return np.hstack(vecs)
        return np.array([])
