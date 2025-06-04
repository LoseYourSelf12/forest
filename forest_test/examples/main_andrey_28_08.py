from core import Vectorizer, RandomForestMixer, register_detector, Dataset
from yolo.olga import Kat_28_08 as Yolo28_08Base
from ocr.andrey import Kat_28_08 as Ocr28_08Base

@register_detector("yolo_28_08")
class Yolo28_08(Yolo28_08Base):
    pass

@register_detector("ocr_28_08")
class Ocr28_08(Ocr28_08Base):
    pass

CONFIG = {
    "detectors": ["yolo_28_08", "ocr_28_08"],
    "dataset_csv": "path/to/labels_28_08.csv",
    "mixer_params": {"n_estimators": 100, "random_state": 42}
}

if __name__ == "__main__":
    vectorizer = Vectorizer(CONFIG["detectors"])
    mixer = RandomForestMixer(**CONFIG["mixer_params"])
    ds = Dataset(CONFIG["dataset_csv"])
    for img_path, label in ds:
        local = {"img": None, "txt": ""}
        x = vectorizer(local)
        print("Sample vector", x)
        break
