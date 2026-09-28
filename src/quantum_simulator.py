from __future__ import annotations

import numpy as np

from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler


class QuantumThreatSampler:
    """
    Quantum-circuit simulator for security feature vectors.

    The 14-dimensional security vector is padded to 16 dimensions
    and amplitude-encoded using 4 qubits because:

        2^4 = 16

    This is a simulation on a classical computer, not execution
    on a physical quantum processor.
    """

    def __init__(
        self,
        shots: int = 512
    ) -> None:

        if shots <= 0:
            raise ValueError(
                "shots must be greater than zero."
            )

        self.shots = shots
        self.qubits = 4

    # ============================================================
    # PREPARE FEATURE VECTOR
    # ============================================================

    def _prepare_values(
        self,
        values
    ) -> np.ndarray:
        """
        Convert the input feature vector into
        a normalized 16-dimensional amplitude vector.
        """

        values = np.asarray(
            values,
            dtype=float
        ).flatten()

        if values.size == 0:
            raise ValueError(
                "values cannot be empty."
            )

        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # Security features are represented by magnitude.
        values = np.abs(values)

        # --------------------------------------------------------
        # Pad to 16 dimensions.
        # --------------------------------------------------------

        state_size = 2 ** self.qubits

        if values.size > state_size:
            raise ValueError(
                f"At most {state_size} features are supported."
            )

        padded = np.zeros(
            state_size,
            dtype=float
        )

        padded[:values.size] = values

        # --------------------------------------------------------
        # Normalize as quantum amplitudes.
        # --------------------------------------------------------

        norm = np.linalg.norm(
            padded
        )

        if norm == 0:

            padded[:] = (
                1.0 /
                np.sqrt(state_size)
            )

        else:

            padded = padded / norm

        return padded

    # ============================================================
    # CREATE QUANTUM CIRCUIT
    # ============================================================

    def create_circuit(
        self,
        values
    ) -> QuantumCircuit:
        """
        Create a 4-qubit amplitude-encoded circuit.
        """

        amplitudes = self._prepare_values(
            values
        )

        qc = QuantumCircuit(
            self.qubits
        )

        # Load the complete normalized
        # feature vector into the quantum state.
        qc.initialize(
            amplitudes,
            range(self.qubits)
        )

        qc.measure_all()

        return qc

    # ============================================================
    # SAMPLE QUANTUM STATE
    # ============================================================

    def sample(
        self,
        values
    ) -> dict:
        """
        Execute the quantum circuit using Qiskit's
        statevector sampler and return measurement counts.
        """

        circuit = self.create_circuit(
            values
        )

        sampler = StatevectorSampler()

        result = sampler.run(
            [circuit],
            shots=self.shots
        ).result()

        counts = (
            result[0]
            .data
            .meas
            .get_counts()
        )

        return dict(counts)

    # ============================================================
    # CONVERT COUNTS TO PROBABILITIES
    # ============================================================

    def sample_probabilities(
        self,
        values
    ) -> dict:
        """
        Return normalized measurement probabilities.
        """

        counts = self.sample(
            values
        )

        total = sum(
            counts.values()
        )

        if total == 0:
            return {}

        return {
            state: count / total
            for state, count
            in counts.items()
        }