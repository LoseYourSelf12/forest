from core import Vectorizer, RandomForestMixer, Dataset
from utils.pipeline_utils import prepare_local
import numpy as np
import joblib

CONFIG = {
    "detectors": ["yolo_kat_05_03", "ocr_kat_05_03"],
    "dataset_csv": "path/to/labels_05_03.csv",
    "output_model": "mixer_05_03.joblib",
}

if __name__ == "__main__":
    vectorizer = Vectorizer(CONFIG["detectors"])
    ds = Dataset(CONFIG["dataset_csv"])
    X, y = [], []
    for img_path, labels, _, _ in ds:
        local = prepare_local(img_path)
        vec = vectorizer(local)
        X.append(vec)
        y.append(1 if labels else 0)
    X = np.array(X)
    y = np.array(y)
    mixer = RandomForestMixer(n_estimators=100, random_state=42)
    mixer.fit(X, y)
    mixer.save(CONFIG["output_model"])
