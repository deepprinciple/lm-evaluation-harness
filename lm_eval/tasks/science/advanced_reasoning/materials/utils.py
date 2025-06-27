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


def process_polymer_tg(dataset):
    return dataset.map(lambda x: {
        "SMILES": x["Question"].split(",")[0].strip().strip('"').strip("*"),
        "Density": float(x["Question"].split(",")[1].strip()),
        "Tg": float(x["Answer"]),
        "question": f"SMILES: {x['Question'].split(',')[0].strip()} \n Density: {x['Question'].split(',')[1].strip()} \n",
        "answer": str(x["Answer"])
    })



