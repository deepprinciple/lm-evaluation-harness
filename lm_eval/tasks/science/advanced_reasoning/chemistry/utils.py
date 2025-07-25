from functools import partial

def process_docs(dataset, task):
    """Filter dataset by specific task type"""
    return dataset.filter(lambda x: x["Task"] == task)

# ['Forward Reaction Prediction', 'Retrosynthesis', 'Experimental Techniques', 'Molecular Property']
process_forward_reaction_prediction = partial(process_docs, task='Forward Reaction Prediction')
process_retrosynthesis = partial(process_docs, task='Retrosynthesis')
process_experimental_techniques = partial(process_docs, task='Experimental Techniques')
process_molecular_property = partial(process_docs, task='Molecular Property')
process_quantum_software_usage = partial(process_docs, task='Quantum Software Usage')