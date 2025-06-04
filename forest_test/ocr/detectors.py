from core import register_detector
from .ocrdetector import OcrDetector

# OCR detectors with inline keyword lists

class Kat_05_03(OcrDetector):
    def __init__(self):
        super().__init__(
            name="05.03 Информационная продукция",
            viptexts=["информационная продукция"],
            texts=[],
        )


class Kat_28_08(OcrDetector):
    def __init__(self):
        super().__init__(
            name="28.08 Застройщик",
            viptexts=["застройщик"],
            texts=[],
        )


# Additional OCR detectors can be added here

register_detector("ocr_kat_05_03")(Kat_05_03)
register_detector("ocr_kat_28_08")(Kat_28_08)
