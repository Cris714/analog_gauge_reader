#!/usr/bin/env python3
"""
ocr_evaluation.py

Comprehensive OCR Evaluation Module for Analog Gauges:
  1. Robust Token Matching: Regex, numeric equivalence, OCR confusion resolution,
     and character-level Levenshtein edit similarity.
  2. Spatial Bipartite/IoU Matching: Pairs predicted text boxes with ground truth.
  3. Image-Level Metrics:
       - Net OCR Quality (Desempeño Neto): Accuracy on detected text without penalizing misses.
       - Total OCR Score (Desempeño General): Accuracy penalizing missed detections as 0%.
       - Spatial Detection Rate (Recall): Percentage of ground truth text boxes localized.
  4. Global Aggregation: Computes mean, median (Q2), Q1, Q3, IQR, min, max across all images.
"""

import re
import json
import argparse
from difflib import SequenceMatcher
import numpy as np


def normalize_ocr_text(text):
    """
    Cleans and normalizes OCR tokens:
    - Strips leading/trailing spaces, quotes, and punctuation.
    - Converts comma decimals to dots (e.g. '0,5' -> '0.5').
    - Converts to lowercase.
    """
    if text is None:
        return ""
    s = str(text).strip().lower()
    s = s.replace(",", ".")
    s = s.strip(" '\"`.-:;|/")
    return s


def compare_ocr_tokens(pred, true, is_unit=False):
    """
    Robust comparison between predicted OCR string and ground truth string.
    Returns float similarity score in [0.0, 1.0].
    
    Stages:
      1. Normalization & exact equality check -> 1.0.
      2. Numeric float equivalence (e.g. '10.0' vs '10') -> 1.0.
      3. OCR optical character confusion correction (O->0, l/I->1, S->5, B->8) -> 1.0.
      4. Regex number extraction: if true is numeric and pred contains it -> 1.0.
      5. Substring / regex unit extraction -> 1.0.
      6. Character-level SequenceMatcher ratio for partial credit (e.g. 1 char typo).
    """
    p_norm = normalize_ocr_text(pred)
    t_norm = normalize_ocr_text(true)

    if not p_norm or not t_norm:
        return 0.0

    if p_norm == t_norm:
        return 1.0

    # 1. Numeric float comparison
    try:
        if abs(float(p_norm) - float(t_norm)) < 1e-4:
            return 1.0
    except ValueError:
        pass

    # 2. OCR optical character confusion correction for digits
    ocr_fix = p_norm.translate(str.maketrans("olisb", "01158"))
    try:
        if abs(float(ocr_fix) - float(t_norm)) < 1e-4:
            return 1.0
    except ValueError:
        pass

    # 3. Regex number extraction (e.g. '10bar' -> 10)
    try:
        t_val = float(t_norm)
        nums = re.findall(r"[-+]?\d*\.?\d+", p_norm)
        for n_str in nums:
            try:
                if abs(float(n_str) - t_val) < 1e-4:
                    return 1.0
            except ValueError:
                pass
    except ValueError:
        # True is a unit or text token: check clean substring/word match
        if t_norm in p_norm:
            return 1.0

    # 4. Character-level edit similarity (SequenceMatcher ratio in [0, 1])
    ratio = SequenceMatcher(None, p_norm, t_norm).ratio()
    return round(float(ratio), 4)


def bbox_iou(boxA, boxB):
    """
    Computes IoU between two bounding boxes with keys 'x', 'y', 'width', 'height'.
    """
    xA = max(boxA['x'], boxB['x'])
    yA = max(boxA['y'], boxB['y'])
    xB = min(boxA['x'] + boxA['width'], boxB['x'] + boxB['width'])
    yB = min(boxA['y'] + boxA['height'], boxB['y'] + boxB['height'])

    inter_width = max(0.0, xB - xA)
    inter_height = max(0.0, yB - yA)
    inter_area = inter_width * inter_height

    boxA_area = boxA['width'] * boxA['height']
    boxB_area = boxB['width'] * boxB['height']
    union_area = boxA_area + boxB_area - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def evaluate_ocr_image_items(gt_items, pred_items, iou_threshold=0.35, is_unit=False):
    """
    Evaluates OCR items for a single image modality (numbers or units).
    Returns dictionary with:
      - n_gt: count of ground truth items
      - n_detected: count of items localized with IoU >= iou_threshold
      - detection_rate: n_detected / n_gt
      - net_quality: average score of detected items (no penalty for misses)
      - total_score: average score over all ground truth items (misses count as 0.0)
      - item_details: list of individual matching records
    """
    n_gt = len(gt_items)
    if n_gt == 0:
        return {
            "n_gt": 0,
            "n_detected": 0,
            "detection_rate": 1.0 if len(pred_items) == 0 else 0.0,
            "net_quality": 1.0 if len(pred_items) == 0 else 0.0,
            "total_score": 1.0 if len(pred_items) == 0 else 0.0,
            "details": []
        }

    matched_preds = set()
    details = []
    detected_scores = []
    all_scores = []

    for gt in gt_items:
        gt_text = gt.get("text", "")
        best_iou = 0.0
        best_pred_idx = None

        for idx, pred in enumerate(pred_items):
            if idx in matched_preds:
                continue
            iou = bbox_iou(gt, pred)
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = idx

        if best_iou >= iou_threshold and best_pred_idx is not None:
            matched_preds.add(best_pred_idx)
            pred_match = pred_items[best_pred_idx]
            pred_text = pred_match.get("text", str(pred_match.get("number", "")))
            score = compare_ocr_tokens(pred_text, gt_text, is_unit=is_unit)
            detected_scores.append(score)
            all_scores.append(score)
            details.append({
                "gt_text": gt_text,
                "pred_text": pred_text,
                "score": score,
                "iou": round(best_iou, 3),
                "detected": True
            })
        else:
            all_scores.append(0.0)
            details.append({
                "gt_text": gt_text,
                "pred_text": None,
                "score": 0.0,
                "iou": round(best_iou, 3),
                "detected": False
            })

    n_detected = len(detected_scores)
    net_quality = float(np.mean(detected_scores)) if n_detected > 0 else 0.0
    total_score = float(np.sum(all_scores) / n_gt)
    detection_rate = float(n_detected / n_gt)

    return {
        "n_gt": n_gt,
        "n_detected": n_detected,
        "detection_rate": round(detection_rate, 4),
        "net_quality": round(net_quality, 4),
        "total_score": round(total_score, 4),
        "details": details
    }


def compute_global_statistics(values_list):
    """
    Computes robust position and central tendency statistics:
    Mean, Median (Q2), Q1, Q3, IQR, Min, Max.
    """
    if not values_list:
        return {}
    arr = np.array(values_list, dtype=float)
    q1 = float(np.percentile(arr, 25))
    q2 = float(np.percentile(arr, 50))
    q3 = float(np.percentile(arr, 75))
    return {
        "mean": round(float(np.mean(arr)), 4),
        "median": round(q2, 4),
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "iqr": round(q3 - q1, 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "pct_gte_80": round(float(np.mean(arr >= 0.8) * 100), 1),
        "pct_gte_95": round(float(np.mean(arr >= 0.95) * 100), 1)
    }


def evaluate_single_image_ocr(gt_numbers, gt_units, pred_numbers, pred_units, iou_threshold=0.35):
    """
    Evaluates both numbers and units for a single image in dial coordinate space.
    Returns:
      - numbers_eval: evaluation dict for scale numbers
      - units_eval: evaluation dict for scale units
      - combined: combined net quality and total score across numbers and units
    """
    # Sanitize and ensure clean list of dicts
    if not isinstance(pred_numbers, list):
        pred_numbers = [pred_numbers] if isinstance(pred_numbers, dict) else []
    if not isinstance(pred_units, list):
        pred_units = [pred_units] if isinstance(pred_units, dict) else []

    if not isinstance(gt_numbers, list):
        gt_numbers = [gt_numbers] if isinstance(gt_numbers, dict) else []
    if not isinstance(gt_units, list):
        gt_units = [gt_units] if isinstance(gt_units, dict) else []

    pred_numbers = [p for p in pred_numbers if isinstance(p, dict) and "x" in p]
    pred_units = [p for p in pred_units if isinstance(p, dict) and "x" in p]
    gt_numbers = [g for g in gt_numbers if isinstance(g, dict) and "x" in g]
    gt_units = [g for g in gt_units if isinstance(g, dict) and "x" in g]

    num_eval = evaluate_ocr_image_items(gt_numbers, pred_numbers, iou_threshold=iou_threshold, is_unit=False)
    unit_eval = evaluate_ocr_image_items(gt_units, pred_units, iou_threshold=iou_threshold, is_unit=True)

    # Combined metrics
    total_gt = num_eval["n_gt"] + unit_eval["n_gt"]
    total_detected = num_eval["n_detected"] + unit_eval["n_detected"]
    
    all_details = num_eval["details"] + unit_eval["details"]
    detected_scores = [d["score"] for d in all_details if d["detected"]]
    all_scores = [d["score"] for d in all_details]

    comb_net = float(np.mean(detected_scores)) if total_detected > 0 else 0.0
    comb_total = float(np.sum(all_scores) / total_gt) if total_gt > 0 else 0.0
    comb_det_rate = float(total_detected / total_gt) if total_gt > 0 else 0.0

    return {
        "numbers": num_eval,
        "units": unit_eval,
        "combined": {
            "n_gt": total_gt,
            "n_detected": total_detected,
            "detection_rate": round(comb_det_rate, 4),
            "net_quality": round(comb_net, 4),
            "total_score": round(comb_total, 4)
        }
    }

