from datasets import Dataset
from sklearn.metrics import f1_score
from collections import defaultdict
import logging
import rdkit
from rdkit import Chem

def process_mof_water_stability(dataset):
    # Rename columns to match doc_to_text variables
    processed = []
    for ex in dataset:
        processed.append({
            "question": ex["Question:"],
            "option_a": ex["Option_A"],
            "option_b": ex["Option_B"],
            "option_c": ex["Option_C"],
            "answer": ex["Answer"]
        })
    return Dataset.from_list(processed)

def process_pxrd_lattice(dataset):
    def format_row(row):
        return {
            "pxrd": row["Peak positions"].strip(),
            "miller_indices": row["Miller indices"].strip(),
            "crystal_system": row["Crystal system"].strip(),
            "comment": row["Comment"].strip(),  # e.g., "MOF" or "ionic compound"
            "target": f"{row['Ground truth-a']}, {row['Ground truth-b']}, {row['Ground truth-c']}"
        }
    return dataset.map(format_row)

def process_pxrd_crystal_system(dataset):
    def format_row(row):
        return {
            "peaks": row["Peak positions"].strip(),
            "intensities": row["Peak intensities"].strip(),
            "material_type": row["Comment"].strip(),
            "label": row["Ground truth - Crystal system"].strip().lower()
        }
    return dataset.map(format_row)

def normalize_prediction(doc, pred):
    """Normalize prediction and target for exact match"""
    if isinstance(pred, list):
        pred = pred[0]
    
    # Strip and lowercase both prediction and target
    pred = pred.strip().lower()
    target = doc["label"].strip().lower()
    
    return {"exact_match": pred == target}

def process_corrosion_prediction(dataset):
    def format_row(row):
        return {
            "smiles": row["smiles"].strip(),
            "label": str(row["corrosion_status"]).strip()
        }
    return dataset.map(format_row)


def process_safety_prediction(dataset):
    def format_row(row):
        return {
            "smiles": row["smiles"].strip(),
            "unsafe": row["Unsafe"],   # "TRUE" or "FALSE"
            "comment": row["Comment"].strip(),         # e.g. "Flammable Liquid"
        }
    return dataset.map(format_row)


def process_pxrd_lattice_prediction(results, doc):
    # 使用正则表达式提取三个浮点数
    import re
    pattern = r"(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)"
    match_results = re.search(pattern, results.get("target", ""))
    match_doc = re.search(pattern, doc[0][0])
    if not match_doc:
        return {"acc": 0.0}
    
    # 获取三个浮点数
    a_doc, b_doc, c_doc = match_doc.groups()
    a_results, b_results, c_results = match_results.groups()
    try:
        # 转换为浮点数以验证格式
        float(a_doc), float(b_doc), float(c_doc)
        float(a_results), float(b_results), float(c_results)
    except ValueError:
        return {"acc": 0.0}
    correct = 0
    if abs(float(a_doc) - float(a_results)) < 3:
        correct += 1
    if abs(float(b_doc) - float(b_results)) < 3:
        correct += 1
    if abs(float(c_doc) - float(c_results)) < 3:
        correct += 1
    return {
        "acc": correct / 3
    }
