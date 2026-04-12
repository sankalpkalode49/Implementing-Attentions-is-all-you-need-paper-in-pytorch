# Implementing-Attentions-is-all-you-need-paper-in-pytorch
In this repository I try to implement the Transformer Architecture from the attention is all you need paper from scratch in pytorch.

⚙️ Core Components Implemented
[x] Token Embeddings: Maps vocabulary indices to dense continuous vectors.
[x] Positional Encodings: Constant, non-learned sinusoidal matrices added to embeddings.
[x] Masking Mechanisms: - Padding Mask: Ignores <PAD> tokens during attention calculations.Look-Ahead Mask: Prevents the decoder from cheating by attending to future tokens.
[x] Multi-Head Attention: Splitting embeddings into $h$ heads for parallel attention calculations.[
x] Position-wise Feed-Forward Networks: Two linear transformations with a ReLU activation in between.
[x] Residual Connections & Layer Normalization: Ensures stable gradients during deep network training.[x] 
Final Linear Projection: Maps the decoder output back to vocabulary-sized logits.
🚀 InstallationClone 
the repository and install the required dependencies:
git clone [https://github.com/yourusername/transformer-from-scratch.git](https://github.com/yourusername/transformer-from-scratch.git)
cd transformer-from-scratch
pip install torch numpy


