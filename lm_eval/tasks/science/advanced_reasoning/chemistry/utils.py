"""
Utility functions for processing evaluation datasets.
"""

import re
from typing import Dict, List
from functools import partial

def process_docs(dataset, task):
    """Filter dataset by specific task type"""
    return dataset.filter(lambda x: x["Task"] == task)

process_forward_reaction_prediction = partial(process_docs, task='Forward Reaction Prediction')
process_retrosynthesis = partial(process_docs, task='Retrosynthesis')
process_experimental_techniques = partial(process_docs, task='Experimental Techniques')
process_molecular_property = partial(process_docs, task='Molecular Property')
process_quantum_software_usage = partial(process_docs, task='Quantum Software Usage')

def process_smiles(doc, results):
    try:
        from rdkit import Chem
    except ImportError as e:
        raise ImportError(
            "This evaluation requires RDKit. Please install rdkit via `conda install -c conda-forge rdkit`"
        ) from e

    reference = doc.get("SMILES", "")
    mols = results[0] if results and isinstance(results[0], list) else results

    if not mols:
        return {"acc": 0.0}

    # Always use the first candidate
    smiles = mols[0]
    mol = Chem.MolFromSmiles(smiles)
    ref_mol = Chem.MolFromSmiles(reference)

    if not mol or not ref_mol:
        return {"acc": 0.0}

    # Compare canonical SMILES
    pred_canon = Chem.MolToSmiles(mol, canonical=True)
    ref_canon = Chem.MolToSmiles(ref_mol, canonical=True)

    return {"acc": 1.0 if pred_canon == ref_canon else 0.0}