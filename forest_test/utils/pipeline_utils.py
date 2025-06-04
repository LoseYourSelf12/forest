from typing import Dict, Any
from PIL import Image
import os
from .load_image import load_from_file
from ocr import Reader

_ocr_reader = Reader()

_ocr_cache: Dict[str, str] = {}
_qwen_cache: Dict[str, str] = {}


def prepare_local(img_path: str) -> Dict[str, Any]:
    """Load image and cached texts."""
    img = load_from_file(img_path)
    result = {
        "img": img,
        "file_name": os.path.basename(img_path),
        "current_file_name": img_path,
    }
    if img_path in _ocr_cache:
        result["txt"] = _ocr_cache[img_path]
    else:
        if img is not None:
            content, disclaimer = _ocr_reader.readbanner(img)
            txt = " ".join(content) + " " + disclaimer
        else:
            txt = ""
        _ocr_cache[img_path] = txt
        result["txt"] = txt
    if img_path in _qwen_cache:
        result["qwen_text"] = _qwen_cache[img_path]
    # Qwen text will be loaded by detector if not present
    return result
