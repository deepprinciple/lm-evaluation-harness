from datasets import Dataset

def property_based_matching(dataset):
    def format_row(row):
        return {
            "TARGET_SMILES": (row.get("TARGET_SMILES") or "").strip(),
            "PROPERTY_VALUE": (row.get("PROPERTY_VALUE") or "").strip(),
            "option_A": (row.get("option_A") or "").strip(),
            "option_B": (row.get("option_B") or "").strip(),
            "option_C": (row.get("option_C") or "").strip(),
            "option_D": (row.get("option_D") or "").strip(),
            "Answer": (row.get("Answer") or "").strip()
        }
    return dataset.map(format_row)

def matched_molecular_pair(dataset):
    def format_row(row):
        return {
            "TARGET_SMILES": (row.get("TARGET_SMILES") or "").strip(),
            "PROPERTY_VALUE": (row.get("PROPERTY_VALUE") or "").strip(),
            "COMPARISON_SMILES": (row.get("COMPARISON_SMILES") or "").strip(),
            "Answer": (row.get("Answer") or "").strip()
        }
    return dataset.map(format_row)

def process_descriptor_prediction(doc, results):
    Answer_HBD = str(doc["Answer_HBD"]).strip()
    Answer_HBA = str(doc["Answer_HBA"]).strip()
    Answer_MW = str(doc["Answer_MW"]).strip()
    Answer_LogP = str(doc["Answer_LogP"]).strip()
    answer = results[0][0] if results and results[0] else ""
    parts = [p.strip() for p in answer.split(",")] if answer else []
    # Ensure we have exactly 4 parts, fill missing with empty string
    while len(parts) < 4:
        parts.append("")
    HBD, HBA, MW, LogP = parts[:4]
    
    # Count exact matches
    matches = 0
    if HBD == Answer_HBD:
        matches += 1
    if HBA == Answer_HBA:
        matches += 1
    if MW == Answer_MW:
        matches += 1
    if LogP == Answer_LogP:
        matches += 1
    
    # Calculate accuracy as number of matches divided by 4
    acc = matches / 4.0
    
    return {"acc": acc}

def protein_localization(dataset):
    def format_row(row):
        return {
            "SEQUENCE": (row.get("SEQUENCE") or "").strip(),
            "Answer": (row.get("Answer") or "").strip()
        }
    return dataset.map(format_row)

def gwas_causal_gene(dataset):
    def format_row(row):
        return {
            "PHENOTYPE": (row.get("PHENOTYPE") or "").strip(),
            "GENE_LIST": (row.get("GENE_LIST") or "").strip(),
            "Answer": (row.get("Answer") or "").strip()
        }
    return dataset.map(format_row)

def process_crispr_delivery(doc, results):
    # Extract option scores and convert to float, handle missing or invalid values gracefully
    option_keys = ['a', 'b', 'c', 'd', 'e', 'f']
    option_scores = {}
    for k in option_keys:
        try:
            option_scores[k] = float(doc.get(f"option_{k}_score", 0.0))
        except (ValueError, TypeError):
            option_scores[k] = 0.0

    # Extract the predicted answers robustly
    Answer = results[0][0] if results and results[0] else ""
    # Split by comma, strip whitespace, lowercase, and filter out empty strings
    answers = [a.strip().lower() for a in Answer.split(",") if a.strip()]
    # Only keep unique, valid options (max 2)
    valid_answers = []
    for a in answers:
        if a in option_scores and a not in valid_answers:
            valid_answers.append(a)
        if len(valid_answers) == 2:
            break
    # If less than 2 valid answers, pad with None
    while len(valid_answers) < 2:
        valid_answers.append(None)

    # Calculate actual score by summing scores of predicted options
    actual_score = 0.0
    for a in valid_answers:
        if a in option_scores:
            actual_score += option_scores[a]

    # Calculate maximum possible score by taking the two highest scores
    sorted_scores = sorted(option_scores.values(), reverse=True)
    max_possible_score = sum(sorted_scores[:2]) if len(sorted_scores) >= 2 else sum(sorted_scores)

    # Calculate accuracy as actual score divided by max possible score
    acc = actual_score / max_possible_score if max_possible_score > 0 else 0.0

    return {"acc": acc}

def process_enzymatic_reaction_prediction(doc, results):
    response = results[0][0] if results and results[0] else ""
    reference = doc["Answer"]
    try:
        from rdkit import Chem
    except ImportError:
        raise ImportError(
            "This evaluation requires RDKit. Please install rdkit via `conda install -c conda-forge rdkit`"
        )
    response_mol = Chem.MolFromSmiles(response)
    reference_mol = Chem.MolFromSmiles(reference)
    if response_mol and reference_mol:
        acc = int(Chem.MolToSmiles(response_mol) == Chem.MolToSmiles(reference_mol))
    else:
        acc = 0.0
    return {"acc": acc}