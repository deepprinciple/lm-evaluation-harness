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

def process_mof_synthesis_qa(dataset):
    def format_row(row):
        return {
            "Question": row["Question"].strip(),
            "Option_A": row["Option_A"].strip(),
            "Option_B": row["Option_B"].strip(),
            "Option_C": row["Option_C"].strip(),
            "Option_D": row["Option_D"].strip(),
            "Option_E": row.get("Option_E", "").strip(),
            "Answer": row["Answer"].strip()  # no XML tags!
        }
    return dataset.map(format_row)

