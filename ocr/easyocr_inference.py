import os
import sys
import numpy as np
import cv2
import torch
import easyocr

# Append path of parent directory to system to import all modules correctly
parent_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from ocr.ocr_reading import OCRReading, UNIT_LIST

_EASYOCR_READER = None


def get_easyocr_reader(gpu=None):
    """
    Returns a cached EasyOCR Reader singleton.
    """
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        if gpu is None:
            gpu = torch.cuda.is_available()
        _EASYOCR_READER = easyocr.Reader(['en'], gpu=gpu)
    return _EASYOCR_READER


def ocr_easyocr(img,
                visualize=True,
                reader=None,
                allowlist_nums='-0123456789.,',
                text_threshold=0.3,
                low_text=0.2,
                mag_ratio=1.3,
                canvas_size=1280):
    """
    Detect and recognize characters in the image using EasyOCR with allowlist.
    Optimized for dial gauges: CRAFT detection is run once, and recognition
    is performed with allowlist for numbers and raw/dictionary matching for units.

    :param img: numpy image (RGB or BGR)
    :param visualize: bool, if True returns (readings, visualization)
    :param reader: optional pre-instantiated EasyOCR reader
    :param allowlist_nums: character allowlist for numeric scale values
    :param text_threshold: CRAFT text confidence threshold
    :param low_text: CRAFT text low-bound score
    :param mag_ratio: CRAFT image magnification ratio
    :param canvas_size: max image size for CRAFT detector (default 1280 to cap RAM)
    :return: readings list or (readings, visualization)
    """
    if reader is None:
        reader = get_easyocr_reader()

    readings = []
    vis_img = img.copy() if visualize else None

    # Handle RGB vs BGR: EasyOCR expects RGB or grayscale or numpy array
    try:
        h_list, f_list = reader.detect(img,
                                       canvas_size=canvas_size,
                                       text_threshold=text_threshold,
                                       low_text=low_text,
                                       mag_ratio=mag_ratio)
    except Exception as e:
        print(f"EasyOCR detection error: {e}")
        h_list, f_list = [], []

    if not h_list or not h_list[0]:
        if visualize:
            return readings, vis_img
        return readings

    # Run recognition: once raw (for units and general text), once with numeric allowlist
    try:
        raw_res = reader.recognize(img, h_list[0], f_list[0])
        num_res = reader.recognize(img, h_list[0], f_list[0], allowlist=allowlist_nums)
    except Exception as e:
        print(f"EasyOCR recognition error: {e}")
        raw_res, num_res = [], []

    for (raw_box, raw_text, raw_conf), (num_box, num_text, num_conf) in zip(raw_res, num_res):
        polygon = np.array(raw_box, dtype=np.float32)
        raw_clean = str(raw_text).strip()
        num_clean = str(num_text).strip()

        # Check if raw text matches a unit
        is_unit = any(raw_clean.lower() == u.lower() for u in UNIT_LIST)

        if is_unit:
            reading = OCRReading(polygon, raw_clean, float(raw_conf))
            readings.append(reading)
        else:
            # Check if num_clean is a valid number
            num_normalized = OCRReading._clean_number_str(num_clean)
            is_valid_num = False
            try:
                float(num_normalized)
                is_valid_num = True
            except ValueError:
                is_valid_num = False

            if is_valid_num and float(num_conf) > 0.05:
                reading = OCRReading(polygon, num_clean, float(num_conf))
                readings.append(reading)
            else:
                # Fallback: check if raw text itself is a number (e.g. without allowlist)
                raw_normalized = OCRReading._clean_number_str(raw_clean)
                try:
                    float(raw_normalized)
                    reading = OCRReading(polygon, raw_clean, float(raw_conf))
                    readings.append(reading)
                except ValueError:
                    # Neither a unit nor a number, skip or record raw if confident
                    pass

    if visualize:
        # Draw detected readings
        h_img, w_img = vis_img.shape[:2]
        # Invariant font sizing: relative to image height so visual size is consistent across images
        ref_h = 500.0
        scale_factor = h_img / ref_h
        font_scale = max(0.4, 0.6 * scale_factor)
        thickness = max(1, int(round(1.6 * scale_factor)))
        line_thickness = max(1, int(round(2.0 * scale_factor)))

        for r in readings:
            pts = np.int32(r.polygon).reshape((-1, 1, 2))
            if r.is_unit():
                text_color = (255, 0, 255)  # Magenta for unit text
                box_color = (255, 128, 0)   # Sky blue for unit box
                label = f"Unit: {r.reading} ({r.confidence:.2f})"
            elif r.is_number():
                text_color = (0, 255, 0)    # Fluorescent green for numbers
                box_color = (255, 255, 0)   # Bright Cyan for bounding boxes (contrasts with green labels)
                label = f"{r.number:g} ({r.confidence:.2f})"
            else:
                text_color = (0, 255, 255)  # Yellow for other
                box_color = (255, 255, 0)   # Cyan for bounding boxes
                label = f"{r.reading} ({r.confidence:.2f})"

            # Bounding box polygon in box_color (Cyan)
            cv2.polylines(vis_img, [pts], isClosed=True, color=box_color, thickness=line_thickness)
            pt_text = (int(r.center[0]), int(r.center[1]))
            # Text label in text_color (Fluorescent green), no black drop shadow
            cv2.putText(vis_img, label, pt_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        text_color, thickness, cv2.LINE_AA)

        return readings, vis_img

    return readings
