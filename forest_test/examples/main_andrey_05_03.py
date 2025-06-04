from core import Vectorizer, RandomForestMixer, register_detector, Dataset
from yolo.olga import Kat_05_03 as Yolo05_03Base
from ocr.andrey import Kat_05_03 as Ocr05_03Base

@register_detector("yolo_05_03")
class Yolo05_03(Yolo05_03Base):
    pass

@register_detector("ocr_05_03")
class Ocr05_03(Ocr05_03Base):
    pass

CONFIG = {
    "detectors": ["yolo_05_03", "ocr_05_03"],
    "dataset_csv": "path/to/labels_05_03.csv",
    "mixer_params": {"n_estimators": 100, "random_state": 42}
}

if __name__ == "__main__":
    vectorizer = Vectorizer(CONFIG["detectors"])
    mixer = RandomForestMixer(**CONFIG["mixer_params"])
    ds = Dataset(CONFIG["dataset_csv"])
    for img_path, label in ds:
        # Обычно здесь подготавливают 'local' с картинкой и текстом
        local = {"img": None, "txt": ""}
        x = vectorizer(local)
        print("Sample vector", x)
        break
