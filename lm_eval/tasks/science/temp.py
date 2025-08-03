from rdkit import Chem
from rdkit.Chem import Lipinski, Descriptors

# SMILES string
smiles = "CN1CCN(S(=O)(=O)c2ccc3nccc(Nc4ccc(Oc5ccccc5)cc4)c3c2)CC1"

# Create molecule object
mol = Chem.MolFromSmiles(smiles)

if mol is not None:
    # Calculate properties
    hbd = Lipinski.NumHDonors(mol)  # Hydrogen Bond Donors
    hba = Lipinski.NumHAcceptors(mol)  # Hydrogen Bond Acceptors
    mw = Descriptors.MolWt(mol)  # Molecular Weight
    logp = Descriptors.MolLogP(mol)  # LogP
    
    # Format and print output
    print(f"<HBD>{hbd}</HBD>")
    print(f"<HBA>{hba}</HBA>")
    print(f"<MW>{mw:.2f}</MW>")
    print(f"<LogP>{logp:.2f}</LogP>")
else:
    print("Error: Could not parse SMILES string")

