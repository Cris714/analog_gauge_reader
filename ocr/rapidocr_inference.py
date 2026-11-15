import os
import sys
import numpy as np
import cv2

# Append path of parent directory to system to import all modules correctly
parent_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from ocr.ocr_reading import OCRReading, UNIT_LIST

_RAPIDOCR_ENGINE = None


def get_rapidocr_engine():
    """
    Returns a cached RapidOCR singleton instance.
    """
    global _RAPIDOCR_ENGINE
    if _RAPIDOCR_ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        _RAPIDOCR_ENGINE = RapidOCR()
    return _RAPIDOCR_ENGINE


def ocr_rapidocr(img,
                 visualize=True,
                 engine=None,
                 use_cls=True):
    """
    Detect and recognize characters in the image using RapidOCR (PP-OCRv4 ONNX).
    Fast, lightweight, and handles rotated dials using the angle classifier.

    :param img: numpy image (RGB or BGR)
    :param visualize: bool, if True returns (readings, visualization)
    :param engine: optional pre-instantiated RapidOCR engine
    :param use_cls: bool, whether to run angle classification (0/90/180/270 deg)
    :return: readings list or (readings, visualization)
    """
    if engine is None:
        engine = get_rapidocr_engine()

    readings = []
    vis_img = img.copy() if visualize else None

    try:
        results, _ = engine(img, use_cls=use_cls)
    except Exception as e:
        print(f"RapidOCR inference error: {e}")
        results = None

    if not results:
        if visualize:
            return readings, vis_img
        return readings

    for item in results:
        # RapidOCR format: [box, text, score]
        box = item[0]
        text = str(item[1]).strip()
        score = float(item[2])

        polygon = np.array(box, dtype=np.float32)

        reading = OCRReading(polygon, text, score)
        readings.append(reading)

    if visualize:
        for r in readings:
            pts = np.int32(r.polygon).reshape((-1, 1, 2))
            if r.is_unit():
                color = (255, 0, 255)  # Magenta for units
                label = f"Unit: {r.reading} ({r.confidence:.2f})"
            elif r.is_number():
                color = (0, 255, 0)    # Green for numbers
                label = f"{r.number:g} ({r.confidence:.2f})"
            else:
                color = (255, 255, 0)  # Yellow for other text
                label = f"{r.reading} ({r.confidence:.2f})"

            cv2.polylines(vis_img, [pts], isClosed=True, color=color, thickness=2)
            pt_text = (int(r.center[0]), int(r.center[1]))
            cv2.putText(vis_img, label, pt_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(vis_img, label, pt_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        color, 1, cv2.LINE_AA)

        return readings, vis_img

    return readings
