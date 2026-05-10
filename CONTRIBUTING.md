# Contributing to Bio-AI Error Detection

First off, thank you for considering contributing to Synthetix-Bio! It’s people like you that make this tool better for the bioinformatics community.

### How Can I Contribute?

#### Reporting Bugs
* Check the **Issues** tab to see if the bug has already been reported.
* If not, open a new issue using the "Bug Report" template.
* Include your Python version, PyTorch version, and a small snippet of the FASTQ data causing the error.

#### Suggesting Enhancements
* We are specifically looking for improvements in:
    * **Indel Detection**: Tuning the Transformer heads for better gap identification.
    * **Performance**: Optimizing the One-Hot encoding for larger SRA datasets.
    * **Scalability**: Adding Distributed Data Parallel (DDP) support for multi-GPU training.

#### Pull Requests
1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests!
3. Ensure the documentation (README) is updated if you change the model architecture.
4. Issue a Pull Request to the `Synthetix-Bio` main branch.

### Style Guide
* Follow **PEP 8** for Python code.
* Use clear, descriptive commit messages (e.g., `feat: add residual connections to GenomePolisher`).
