from functools import partial

def process_docs(dataset, task):
    return dataset.filter(lambda x: x["Task"] == task)

process_quantum_information = partial(process_docs, task="Quantum Information")
process_dmrg_tensor_networks = partial(process_docs, task="DMRG/Tensor Networks")
