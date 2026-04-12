# Implementing-Attentions-is-all-you-need-paper-in-pytorch
In this repository I try to implement the Transformer Architecture from the attention is all you need paper from scratch in pytorch.
⚙️ Core Components Implemented[x] Token Embeddings: Maps vocabulary indices to dense continuous vectors.[x] Positional Encodings: Constant, non-learned sinusoidal matrices added to embeddings.[x] Masking Mechanisms: - Padding Mask: Ignores <PAD> tokens during attention calculations.Look-Ahead Mask: Prevents the decoder from cheating by attending to future tokens.[x] Multi-Head Attention: Splitting embeddings into $h$ heads for parallel attention calculations.[x] Position-wise Feed-Forward Networks: Two linear transformations with a ReLU activation in between.[x] Residual Connections & Layer Normalization: Ensures stable gradients during deep network training.[x] Final Linear Projection: Maps the decoder output back to vocabulary-sized logits.🚀 InstallationClone the repository and install the required dependencies:Bashgit clone [https://github.com/yourusername/transformer-from-scratch.git](https://github.com/yourusername/transformer-from-scratch.git)
cd transformer-from-scratch
pip install torch numpy
💻 UsageHere is a quick example of how to initialize the full Transformer model and pass a batch of sequences through it.Pythonimport torch
from models import Transformer

# Hyperparameters
src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1

# Initialize Model
model = Transformer(
    src_vocab_size, 
    tgt_vocab_size, 
    d_model, 
    num_heads, 
    num_layers, 
    d_ff, 
    max_seq_length, 
    dropout
)

# Dummy inputs (Batch Size: 32, Sequence Length: 10)
src_data = torch.randint(1, src_vocab_size, (32, 10))
tgt_data = torch.randint(1, tgt_vocab_size, (32, 10))

# Forward pass
output = model(src_data, tgt_data)
print(f"Output shape: {output.shape}") 
# Expected shape: [32, 10, 5000] -> [Batch, Seq_Len, Target_Vocab]
