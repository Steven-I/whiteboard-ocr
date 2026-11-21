import os
import time
import shutil
import cv2
import numpy as np
import pytesseract
import easyocr
from PIL import Image
from google import genai
from google.genai import types

# Global singleton state for lazy initialization
_EASYOCR_READER = None

# Explicit Tesseract binary lookup for Windows default installation path
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path


def is_tesseract_available() -> bool:
    """Checks system PATH or explicit path for native Tesseract OCR installation."""
    if os.path.exists(tesseract_path):
        return True
    return shutil.which("tesseract") is not None


def get_easyocr_reader():
    """Lazily initializes CPU-bound EasyOCR reader instance."""
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        _EASYOCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _EASYOCR_READER


def parse_with_baseline(
    gray_img: np.ndarray, 
    binary_img: np.ndarray, 
    boxes: list[tuple[int, int, int, int]]
) -> tuple[str, list[dict], float]:
    """
    LO 4 Baseline Engine: PyTesseract region-by-region extraction.
    """
    start_time = time.time()
    extracted_lines = []
    metadata = []

    if not is_tesseract_available():
        return "Baseline (Tesseract) binary not found on system PATH.", [], 0.0

    for (x, y, w, h) in boxes:
        pad = 5
        y1, y2 = max(0, y - pad), min(binary_img.shape[0], y + h + pad)
        x1, x2 = max(0, x - pad), min(binary_img.shape[1], x + w + pad)
        roi_bin = binary_img[y1:y2, x1:x2]

        if roi_bin.size == 0:
            continue

        try:
            config = '--oem 3 --psm 7'
            text = pytesseract.image_to_string(roi_bin, config=config).strip()
            if text:
                extracted_lines.append(text)
                metadata.append({"box": (x, y, w, h), "text": text, "conf": 0.70})
        except Exception:
            continue

    elapsed = time.time() - start_time
    full_output = "\n".join(extracted_lines) if extracted_lines else "No text extracted by baseline."
    return full_output, metadata, elapsed


def parse_with_advanced(
    gray_img: np.ndarray, 
    binary_img: np.ndarray, 
    boxes: list[tuple[int, int, int, int]],
    gemini_api_key: str | None = None,
    mode: str = "Standard Notes"
) -> tuple[str, list[dict], float, str]:
    """
    LO 4 Advanced Engine: Dual-Track Deep Learning Parser.
    - Track 1: EasyOCR deep feature extraction on preprocessed crops.
    - Track 2: Vision-Language Model (Gemini 2.5 Flash) for complex math/diagrams.
    """
    start_time = time.time()
    metadata = []
    engine_name = "EasyOCR (Deep Features)"

    # Track 2: Vision-Language API execution if key is provided
    if gemini_api_key and gemini_api_key.strip():
        try:
            client = genai.Client(api_key=gemini_api_key.strip())
            rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(rgb_img)

            prompt = (
                "Transcribe all handwritten whiteboard text, math equations, code blocks, and diagrams accurately. "
                "Output clean Markdown. Convert mathematical notation to LaTeX format ($...$ or $$...$$)."
            )
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[pil_img, prompt],
                config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=2048)
            )
            elapsed = time.time() - start_time
            return response.text if response.text else "No text returned.", [], elapsed, "VLM Vision Engine"
        except Exception as e:
            engine_name = f"EasyOCR (Fallback: {str(e)[:30]})"

    # Track 1: EasyOCR Region Extraction
    reader = get_easyocr_reader()
    extracted_lines = []

    if not boxes:
        elapsed = time.time() - start_time
        return "No text regions detected.", [], elapsed, engine_name

    for (x, y, w, h) in boxes:
        pad = 6
        y1, y2 = max(0, y - pad), min(gray_img.shape[0], y + h + pad)
        x1, x2 = max(0, x - pad), min(gray_img.shape[1], x + w + pad)
        roi_gray = gray_img[y1:y2, x1:x2]

        if roi_gray.size == 0:
            continue

        res = reader.readtext(roi_gray, detail=1)
        for (_, text, conf) in res:
            text = text.strip()
            if text:
                extracted_lines.append(text)
                metadata.append({"box": (x, y, w, h), "text": text, "conf": float(conf)})

    elapsed = time.time() - start_time
    full_output = "\n".join(extracted_lines) if extracted_lines else "No text extracted by advanced engine."
    return full_output, metadata, elapsed, engine_name