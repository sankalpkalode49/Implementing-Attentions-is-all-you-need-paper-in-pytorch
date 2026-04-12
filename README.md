# Transformer from Scratch: Attention Is All You Need

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

A clean, modular, and highly readable implementation of the original Transformer architecture as proposed in the seminal 2017 paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) by Vaswani et al. 

This repository is built from the ground up to serve as an educational resource and a foundational codebase for understanding sequence-to-sequence modeling, self-attention mechanisms, and advanced deep learning architectures.

---

## 📖 Table of Contents
* [Architecture Overview](#architecture-overview)
* [Project Structure](#project-structure)
* [Core Components Implemented](#core-components-implemented)
* [Installation](#installation)
* [Usage](#usage)
* [Future Work](#future-work)
* [Author](#author)

---

## 🧠 Architecture Overview

The Transformer relies entirely on an attention mechanism to draw global dependencies between input and output, dispensing with recurrence and convolutions entirely. 

Key innovations implemented in this repository:
1. **Scaled Dot-Product Attention:** The mathematical engine that computes alignment scores.
2. **Multi-Head Attention:** Allows the model to jointly attend to information from different representation subspaces.
3. **Positional Encoding:** Injects sequence order information using sine and cosine functions of different frequencies.

---

## 📁 Project Structure

The codebase is strictly modularized to separate the mathematical operations from the high-level layer stacking.

```text
transformer-from-scratch/
│
├── embeddings.py       # Token Embedding and Positional Encoding classes
├── attention.py        # Scaled Dot-Product and Multi-Head Attention mechanisms
├── layers.py           # FFN, AddNorm, EncoderLayer, and DecoderLayer definitions
├── models.py           # Full Encoder, Decoder, and the overarching Transformer class
├── utils.py            # Helper functions for Padding and Look-ahead masks
├── train.py            # (WIP) Training loop and dataset handling
└── README.md
