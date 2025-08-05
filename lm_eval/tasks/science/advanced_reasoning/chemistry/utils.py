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
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.Chem import DataStructs
    except ImportError as e:
        raise ImportError(
            "This evaluation requires RDKit. Please install rdkit via `conda install -c conda-forge rdkit`"
        ) from e
      
    reference = doc.get("Answer", "")
    mols = results[0] if results and isinstance(results[0], list) else results

    if not mols:
        return {"acc": 0.0}

    # Always use the first candidate
    smiles = mols[0]
    mol = Chem.MolFromSmiles(smiles)
    ref_mol = Chem.MolFromSmiles(reference)
    if not mol or not ref_mol:
        return {"acc": 0.0}

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    # Compute Tanimoto similarity between Morgan fingerprints
    fp_pred = mfpgen.GetFingerprint(mol)
    fp_ref = mfpgen.GetFingerprint(ref_mol)
    tanimoto = DataStructs.TanimotoSimilarity(fp_pred, fp_ref)

    return {"acc": tanimoto}
