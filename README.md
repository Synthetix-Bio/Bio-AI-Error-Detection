# Bio-AI Error Detection System

Developed by **Synthetix-Bio**, this repository hosts a hybrid deep learning framework designed for high-accuracy polishing of Oxford Nanopore Technology (ONT) genomic data.

<img width="562" height="455" alt="download (6)" src="https://github.com/user-attachments/assets/a68ba9d8-2277-4ddf-ae89-2b23abe7d768" />


---

## Executive Summary

Third-generation sequencing, particularly ONT, offers long-read capabilities but is often limited by systematic error profiles, especially in homopolymeric regions. This project implements a **Hybrid CNN-Transformer** architecture to detect and correct these errors by combining local motif extraction with global sequence context.

---

## Architecture Overview

The core of the system is the `GenomePolisher` model, which utilizes a dual-stage feature extraction process:

### 1. Local Motif Extraction (CNN)

* **1D-Convolutional Layer**: Uses a kernel size of 5 to capture local k-mer patterns and base-to-base transitions.
* **Batch Normalization**: Stabilizes learning by normalizing internal feature maps.

### 2. Global Contextual Analysis (Transformer)

* **Multi-Head Self-Attention**: 8 attention heads allow the model to cross-reference distant bases across a 100bp window, identifying long-range dependencies that HMM-based polishers often miss.
* **Positional Encoding**: Learnable embeddings provide essential spatial awareness for the 5' to 3' DNA directionality.

---

## Key Features

* **Multi-Modal Preprocessing**: Converts raw FASTQ files into One-Hot Encoded NumPy matrices (Shape: $Length \times 4$).
* **Quality Awareness**: Integrated Phred quality score analysis to visualize and filter low-confidence reads before training.
* **Advanced Error Profiling**: Built-in visualization tools for:
* **Homopolymer Length Correlation Heatmaps** to track systematic insertion/deletion (Indel) bias.
* **Nucleotide Prediction Confusion Matrices** (A, C, G, T, and Gap).
* **Training vs. Validation curves** to monitor for overfitting.



---

## Methods and Data

* **Data Sources**: Benchmarked using SRA datasets including Zymo, Human, and E. coli.
* **Training Objective**: Cross-Entropy Loss with AdamW optimization and gradient clipping to ensure stable Transformer convergence.
* **Encoding**: DNA sequences are standardized to uppercase and mapped to a 4-channel vector; 'N' or unknown bases are treated as null entries.

---

## Getting Started

### Prerequisites

```bash
pip install biopython matplotlib numpy torch scikit-learn seaborn

```

### Basic Usage

1. **Analyze Quality**:
```python
from bio_ai_error_detection_system import analyze_ont_quality
analyze_ont_quality('your_data.fastq')

```


2. **Train the Polisher**:
```python
model = GenomePolisher(seq_len=100, in_channels=4, num_classes=5)
trained_model = train_polisher(model, your_dataloader)

```



---

## Conclusion

By leveraging the local sensitivity of CNNs and the global reasoning of Transformers, this system achieves superior k-mer completeness and significant reductions in indel rates compared to traditional HMM-based tools like Racon.

---

## ## License

This project is released under the MIT License by **Synthetix-Bio**.
