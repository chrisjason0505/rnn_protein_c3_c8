# Protein Sequence Prediction using RNNs 

This project explores the use of Recurrent Neural Network (RNN) based models to predict protein sequence classes, focusing on the **C3** and **C8** classification tasks.  
The project is currently **under active development** as part of an academic assignment.

## Overview

Protein sequences are sequential by nature, making them suitable for sequence modeling techniques such as RNNs. This project aims to understand how well RNN-based architectures can capture patterns in amino acid sequences and use them for classification.

## Tasks

- **C3** — 3-class protein sequence classification  
- **C8** — 8-class protein sequence classification  

Each task predicts a label for a given protein sequence.

## Approach

1. Encode amino acid sequences into numerical representations.
2. Train RNN-based models (RNN, LSTM, BiLSTM).
3. Evaluate performance on C3 and C8 tasks.
4. Analyze results and improve the model iteratively.

## Models

- RNN  
- LSTM  
- Bidirectional LSTM (BiLSTM)

## Tech Stack

- Python  
- PyTorch / TensorFlow  
- NumPy, Pandas  
- Scikit-learn

## Status

🚧 **This project is a work in progress.**  
Models, results, and documentation may change as experiments continue.

## How to Run

```bash
pip install -r requirements.txt
python train.py
python evaluate.py
