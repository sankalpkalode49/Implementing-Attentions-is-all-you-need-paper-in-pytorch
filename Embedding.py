import torch
import torch.nn as nn


import math

class Token_Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        self.embed = nn.Embedding(vocab_size, d_model)

        self.d_model = d_model
    
    def forword(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: input sequence of token ID: [batch_size,seq_length]
        return embedded sequence shape [batch_size,seq_length,d_model]

        """
        return self.embed(x) * math.sqrt(self.d_model)

class Positional_Embedding(nn.Module):
    def __int__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        super.__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_length, d_model)

        # Create a column vector of positions: [[0], [1], [2], ..., [max_seq_length - 1]]
        # Shape: [max_seq_length, 1]
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        # Calculate the division term using log space for numerical stability
        # Shape: [d_model / 2]

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices (0, 2, 4...)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices (1, 3, 5...)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension so it becomes [1, max_seq_length, d_model]
        # This allows it to broadcast across batch sizes during the forward pass
        pe = pe.unsqueeze(0)
        
        # register_buffer saves 'pe' to the model's state_dict but tells PyTorch 
        # NOT to treat it as a learnable parameter (the optimizer won't touch it)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token embeddings. Shape: [batch_size, seq_length, d_model]
        Returns:
            Embeddings with positional encodings added.
        """
        # Slice 'pe' to match the actual sequence length of the input 'x'
        # x.size(1) gets the seq_length dimension
        x = x + self.pe[:, :x.size(1), :]
        
        return self.dropout(x)
    

    



class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of the 
    tokens in the sequence.
    """
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        super().__init__()
        # The paper applies dropout to the sum of embeddings and positional encodings
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of shape (max_seq_length, d_model) initialized with zeros
        pe = torch.zeros(max_seq_length, d_model)
        
        # Create a column vector of positions: [[0], [1], [2], ..., [max_seq_length - 1]]
        # Shape: [max_seq_length, 1]
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Calculate the division term using log space for numerical stability
        # Shape: [d_model / 2]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices (0, 2, 4...)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices (1, 3, 5...)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension so it becomes [1, max_seq_length, d_model]
        # This allows it to broadcast across batch sizes during the forward pass
        pe = pe.unsqueeze(0)
        
        # register_buffer saves 'pe' to the model's state_dict but tells PyTorch 
        # NOT to treat it as a learnable parameter (the optimizer won't touch it)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token embeddings. Shape: [batch_size, seq_length, d_model]
        Returns:
            Embeddings with positional encodings added.
        """
        # Slice 'pe' to match the actual sequence length of the input 'x'
        # x.size(1) gets the seq_length dimension
        x = x + self.pe[:, :x.size(1), :]
        
        return self.dropout(x)
