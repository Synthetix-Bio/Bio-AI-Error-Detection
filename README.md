# Bio-AI-Error-Detection

GenomePolisher is a Python bioinformatics project for DNA sequence error correction in Oxford Nanopore Technology (ONT) reads.

## CNN-Transformer Genomic Polishing Approach

The model combines local pattern extraction and long-range context modeling:

- **1D-CNN backbone** learns local sequencing error signatures (insertions, deletions, substitutions) from one-hot encoded bases.
- **Multi-head self-attention** captures long-distance dependencies across each read.
- **Base-wise classifier** predicts corrected nucleotide classes (`A/C/G/T`) for each position.

## Project Structure

- `src/model.py`: PyTorch `GenomePolisher` hybrid 1D-CNN + multi-head attention model.
- `src/preprocessing.py`: FASTQ parsing via BioPython and one-hot encoding utilities.
- `src/train.py`: Training pipeline with AdamW optimization and gradient clipping.
- `requirements.txt`: Required Python dependencies.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare ONT FASTQ reads and encode them with `src/preprocessing.py`.
3. Train the model using `train_genome_polisher(...)` from `src/train.py`.

This repository provides a clear baseline scaffold for CNN-Transformer-based genomic polishing workflows.
