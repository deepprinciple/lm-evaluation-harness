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

def lipinski_prediction(dataset):
    def format_row(row):
        return {
            "INPUT_SMILES": (row.get("INPUT_SMILES") or "").strip(),
            "Answer": (row.get("Answer") or "").strip()
        }
    return dataset.map(format_row)

def fragment_completion(dataset):
    def format_row(row):
        return {
            "MASKED_SMILES": (row.get("MASKED_SMILES") or "").strip(),
            "PROPERTY_VALUE": (row.get("PROPERTY_VALUE") or "").strip(),
            "option_A": (row.get("option_A") or "").strip(),
            "option_B": (row.get("option_B") or "").strip(),
            "option_C": (row.get("option_C") or "").strip(),
            "option_D": (row.get("option_D") or "").strip(),
            "Answer": (row.get("Answer") or "").strip()
        }
    return dataset.map(format_row)

def descriptor_prediction(dataset):
    def format_row(row):
        return {
            "INPUT_SMILES": (row.get("INPUT_SMILES") or "").strip(),
            "Answer_HBD": (row.get("Answer_HBD") or "").strip(),
            "Answer_HBA": (row.get("Answer_HBA") or "").strip(),
            "Answer_MW": (row.get("Answer_MW") or "").strip(),
            "Answer_LogP": (row.get("Answer_LogP") or "").strip()
        }
    return dataset.map(format_row)

def process_descriptor_prediction(doc, results):
    Answer_HBD = doc["Answer_HBD"]
    Answer_HBA = doc["Answer_HBA"]
    Answer_MW = doc["Answer_MW"]
    Answer_LogP = doc["Answer_LogP"]
    answer = results[0]
    HBD = answer.split(",")[0]
    HBA = answer.split(",")[1]
    MW = answer.split(",")[2]
    LogP = answer.split(",")[3]
    
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

def enzymatic_reaction_prediction(dataset):
    def format_row(row):
        return {
            "REACTANTS": (row.get("REACTANTS") or "").strip(),
            "EC_NUMBER": (row.get("EC_NUMBER") or "").strip(),
            "SUPPORTING_INFORMATION": (row.get("SUPPORTING_INFORMATION") or "").strip(),
            "SPECIFIC_INSTRUCTIONS": (row.get("SPECIFIC_INSTRUCTIONS") or "").strip(),
            "Answer": (row.get("Answer") or "").strip()
        }
    return dataset.map(format_row)

def gene_editing(dataset):
    def format_row(row):
        return {
            "Question": (row.get("Question") or "").strip(),
            "Option_A": (row.get("Option_A") or "").strip(),
            "Option_B": (row.get("Option_B") or "").strip(),
            "Option_C": (row.get("Option_C") or "").strip(),
            "Option_D": (row.get("Option_D") or "").strip(),
            "Option_E": (row.get("Option_E") or "").strip(),
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

def crispr_delivery(dataset):
    def format_row(row):
        return {
            "USER_MESSAGE": (row.get("USER_MESSAGE") or "").strip(),
            "option_a_score": (row.get("option_a_score") or "").strip(),
            "option_b_score": (row.get("option_b_score") or "").strip(),
            "option_c_score": (row.get("option_c_score") or "").strip(),
            "option_d_score": (row.get("option_d_score") or "").strip(),
            "option_e_score": (row.get("option_e_score") or "").strip(),
            "option_f_score": (row.get("option_f_score") or "").strip()
        }
    return dataset.map(format_row)

def process_crispr_delivery(doc, results):
    # Extract option scores and convert to float
    option_scores = {
        'a': float(doc["option_a_score"]),
        'b': float(doc["option_b_score"]),
        'c': float(doc["option_c_score"]),
        'd': float(doc["option_d_score"]),
        'e': float(doc["option_e_score"]),
        'f': float(doc["option_f_score"])
    }
    
    # Extract the predicted answers
    Answer = results[0]
    Answer1 = Answer.split(",")[0].strip().lower()
    Answer2 = Answer.split(",")[1].strip().lower()
    
    # Calculate actual score by summing scores of predicted options
    actual_score = 0.0
    if Answer1 in option_scores:
        actual_score += option_scores[Answer1]
    if Answer2 in option_scores:
        actual_score += option_scores[Answer2]
    
    # Calculate maximum possible score by taking the two highest scores
    sorted_scores = sorted(option_scores.values(), reverse=True)
    max_possible_score = sorted_scores[0] + sorted_scores[1]
    
    # Calculate accuracy as actual score divided by max possible score
    if max_possible_score > 0:
        acc = actual_score / max_possible_score
    else:
        acc = 0.0
    
    return {"acc": acc}