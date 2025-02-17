import os
import re
import subprocess
from typing import List, Optional
from lm_eval.utils import eval_logger
from datasets import load_dataset
import pandas as pd

def process_smiles_gap(doc, results):
    df =  load_dataset()
    if(df is None):
        eval_logger.info("Dataset loading failed")
        return {
            "mean_homo_lumo_gap": 0.0
        }
    answer = extract_answer(results[0])
    if(answer is None or len(answer) == 0):
        eval_logger.info("Answer extraction failed")
        return {
            "mean_homo_lumo_gap": 0.0
        }
    mean_homo_lumo_gap = evaluate_answer(answer, df, "gap", doc)
    return {
        "mean_homo_lumo_gap": mean_homo_lumo_gap
    }

def process_smiles_polar(doc, results):
    df =  load_dataset()
    if(df is None):
        eval_logger.info("Dataset loading failed")
        return {
            "mean_polarisability": 0.0
        }
    answer = extract_answer(results[0])
    if(answer is None or len(answer) < 5):
        eval_logger.info("Answer extraction failed")
        return {
            "mean_polarisability": 0.0
        }
    mean_polarisability = evaluate_answer(answer, df, "polarisability", doc)
    eval_logger.info("Mean Polarisability: " + str(mean_polarisability))
    return {
        "mean_polarisability": mean_polarisability
    }

def load_dataset():
    url = "https://zenodo.org/records/14328055/files/ground_truth_fitness_values.csv"
    output_directory = "data"
    output_path = os.path.join(output_directory, "ground_truth_fitness_values.csv")

    # make sure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # check if the file exists
    if not os.path.exists(output_path):
        eval_logger.info(f"File '{output_path}' does not exist, starting download...")

        subprocess.run(['wget', url, '-P', output_directory])
        eval_logger.info(f"File downloaded, saved in: {output_path}")
    else:
        eval_logger.info(f"File '{output_path}' already exists, skipping download.")
    
    df = pd.read_csv(output_path)
    return df

def extract_answer(message: str, expected_returns: int = 5):
    """
    Extract TMC strings from LLM response message.
    
    Args:
        message: Response message from LLM
        expected_returns: Expected number of TMCs to extract
        
    Returns:
        List of extracted TMC strings
    """
    # TMC pattern matching regex
    pattern = r"Pd_(\w{6})-subgraph-(\d+)_(\w{6})-subgraph-(\d+)_(\w{6})-subgraph-(\d+)_(\w{6})-subgraph-(\d+)"
    
    # Possible delimiters for TMC in message
    delimiters = [
        "*TMC*", "<<<TMC>>>:", "<TMC>", "TMC:", " TMC"
    ]
    
    # Try to split message using different delimiters
    message_parts = None
    for delimiter in delimiters:
        if delimiter in message:
            message_parts = message.split(delimiter)
            break
            
    if message_parts is None:
        print("Unidentified pattern for splitting the LLM message.")
        return []
    
    # Extract TMCs
    tmcs = []
    for i in range(expected_returns):
        try:
            idx = -expected_returns + i
            match = re.search(pattern, message_parts[idx])
            
            if match:
                tmc = match.group()
                if len(tmc.split("_")) == 5:  # Validate TMC format
                    tmcs.append(tmc)
                else:
                    print(f"Invalid TMC format: {tmc}")
                    
        except IndexError:
            continue
            
    return tmcs

def evaluate_answer(answer, df, porp, doc):
    doc_df = transform_doc_to_df(doc)
    df = find_tmc_in_space(df, answer, doc_df)
    
    if(df is None):
        return 0.0
    else:
        if(len(df) == 5):
            return df[porp].mean()
        elif(len(df) < 5):
            return df[porp].sum() / 5
        else:
            return df[porp][:5].mean()

def transform_doc_to_df(doc):
    given_tmcs = pd.DataFrame(columns=["lig1", "lig2", "lig3", "lig4"])
    for i in range(1, 11):  
        tmc_key = f'tmc{i}'
        if tmc_key in doc:
            tmc = doc[tmc_key].split(';')[0].strip('{}')
            tmc = tmc.split("_")
            tmc = tmc[1:]  # Skip the first element(Pd_)
            tmc = [lig.split('-')[0] for lig in tmc]
            tmc = pd.DataFrame([tmc], columns=["lig1", "lig2", "lig3", "lig4"])
            given_tmcs = pd.concat([given_tmcs, tmc], ignore_index=True)
    return given_tmcs


def find_tmc_in_space(df: pd.DataFrame, tmcs: List[str], doc_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Find TMCs in the search space by checking all possible rotations of ligands.
    
    Args:
        df: DataFrame containing the TMC search space
        tmcs: List of TMC strings to search for
        doc_df: DataFrame containing the given TMCs
    Returns:
        DataFrame containing matched TMCs, or None if no matches found
    """
    matched_tmcs = []
    
    for tmc in tmcs:
        if tmc is None:
            continue
            
        # Get ligands from TMC string
        ligs = tmc.split("_")[1:]
        
        # Check all possible rotational combinations of ligands
        rotations = [
            ligs[i:] + ligs[:i] for i in range(4)
        ]
        
        # Search for each rotation in the DataFrame
        for rot_ligs in rotations:
            match_df = df[
                (df["lig1"] == rot_ligs[0]) &
                (df["lig2"] == rot_ligs[1]) &
                (df["lig3"] == rot_ligs[2]) &
                (df["lig4"] == rot_ligs[3])
            ]

            given_df = doc_df[
                (doc_df["lig1"] == rot_ligs[0]) &
                (doc_df["lig2"] == rot_ligs[1]) &
                (doc_df["lig3"] == rot_ligs[2]) &
                (doc_df["lig4"] == rot_ligs[3])
            ]
            if(len(given_df)):
                eval_logger.info("Given TMC is already in the Given Prompt TMCs, skip this TMC")
                break # if the given TMC is already in the Given Prompt TMCs, skip this TMC
            if len(match_df):
                matched_tmcs.append(match_df)
                break  # Found a match, move to next TMC
                
    return pd.concat(matched_tmcs) if matched_tmcs else None



