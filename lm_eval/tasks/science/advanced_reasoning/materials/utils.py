from datasets import Dataset

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


# For poylmer_tg prediction
def process_polymer_tg(dataset):
    return dataset.map(lambda x: {
        "SMILES": x["SMILES"],
        "Density": x["Density"],
        "Tg": x["Tg"]
    })



'''
Custom Metrics:
'''

# Custom MAE metric for regression tasks
def mae(predictions, targets, **kwargs):
    total_error = 0
    count = 0
    for pred, target in zip(predictions, targets):
        try:
            pred_val = float(pred)
            target_val = float(target)
            total_error += abs(pred_val - target_val)
            count += 1
        except ValueError:
            continue
    return total_error / count if count > 0 else float("inf")
