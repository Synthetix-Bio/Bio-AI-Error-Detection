from typing import List

import numpy as np
from Bio import SeqIO

_BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


def parse_fastq_sequences(fastq_path: str) -> List[str]:
    """Read sequences from a FASTQ file."""
    return [str(record.seq).upper() for record in SeqIO.parse(fastq_path, "fastq")]


def one_hot_encode_sequence(sequence: str, max_length: int | None = None) -> np.ndarray:
    """One-hot encode a DNA sequence into shape (length, 4)."""
    seq = sequence.upper()
    target_length = max_length if max_length is not None else len(seq)
    encoded = np.zeros((target_length, 4), dtype=np.float32)

    for idx, base in enumerate(seq[:target_length]):
        base_index = _BASE_TO_INDEX.get(base)
        if base_index is not None:
            encoded[idx, base_index] = 1.0

    return encoded


def encode_fastq(fastq_path: str, max_length: int | None = None) -> np.ndarray:
    """Parse FASTQ and return one-hot encoded reads with optional padding/truncation."""
    sequences = parse_fastq_sequences(fastq_path)
    if not sequences:
        return np.empty((0, 0, 4), dtype=np.float32)

    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    return np.stack([one_hot_encode_sequence(seq, max_length=max_length) for seq in sequences])
