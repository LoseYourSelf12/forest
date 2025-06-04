from core import Vectorizer, RandomForestMixer, Dataset
from yolo import Kat_05_03
from ocr import Kat_05_03 as Ocr05
from utils.pipeline_utils import prepare_local
import pandas as pd

CONFIG = {
    "detectors": ["yolo_kat_05_03", "ocr_kat_05_03"],
    "dataset_csv": "path/to/labels_05_03.csv",
    "mixer_weights": "mixer_05_03.joblib",
    "output_csv": "results_csv/pred_05_03.csv",
}

if __name__ == "__main__":
    vectorizer = Vectorizer(CONFIG["detectors"])
    mixer = RandomForestMixer()
    mixer.load(CONFIG["mixer_weights"])
    ds = Dataset(CONFIG["dataset_csv"])
    records = []
    for img_path, labels, _, _ in ds:
        local = prepare_local(img_path)
        vec = vectorizer(local).reshape(1, -1)
        pred = mixer.predict(vec)[0]
        records.append({"file_name": img_path, "prediction": pred})
    pd.DataFrame(records).to_csv(CONFIG["output_csv"], index=False)
