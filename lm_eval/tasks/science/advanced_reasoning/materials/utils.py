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

def process_lammps_vasp(dataset):
    def format_row(row):
        return {
            "Question": (row.get("Question") or "").strip(),
            "Option_A": (row.get("Option_A") or "").strip(),
            "Option_B": (row.get("Option_B") or "").strip(),
            "Option_C": (row.get("Option_C") or "").strip(),
            "Option_D": (row.get("Option_D") or "").strip(),
            "Option_E": (row.get("Option_E") or "").strip(),
            "Option_F": (row.get("Option_F") or "").strip(),
            "Answer": (row.get("Answer") or "").strip(),
            "Comment": (row.get("Comment") or "").strip(),  # VASP / LAMMPS
        }
    return dataset.map(format_row)



