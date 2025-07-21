from functools import partial

def process_docs(dataset, task):
    return dataset.filter(lambda x: x["Task"] == task)

process_quantum_information = partial(process_docs, task="Quantum Information")
process_dmrg_tensor_networks = partial(process_docs, task="DMRG/Tensor Network")
process_quantum_compilation_hamiltonian_compilation = partial(process_docs, task="Quantum Compilation, Hamiltonian Compilation")
process_phase_classification_transitions = partial(process_docs, task="Phase Classification & Transitions")
process_general_relativity_cosmology = partial(process_docs, task="General Relativity and Cosmology")
process_condensed_matter_physics = partial(process_docs, task="Condensed Matter Physics")
process_ground_state_discovery = partial(process_docs, task="Ground State Discovery")
process_statistical_mechanics = partial(process_docs, task="Statistical Mechanics")
process_quantum_field_theory = partial(process_docs, task="Quantum Field Theory")
process_amo_quantum_optics = partial(process_docs, task="AMO / Quantum Optics")
process_algebraic_topology = partial(process_docs, task="Algebraic Topology")
process_electromagnetism = partial(process_docs, task="Electromagnetism")
process_quantum_optics = partial(process_docs, task="Quantum Optics")
process_core_knowledge = partial(process_docs, task="Core Knowledge")
