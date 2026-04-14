import torch 
import torch.nn as nn
import math
import numpy  as np
def scaled_dot_product_attention(query, key ,value ,mask = None, dropout = None):
    """
    this function is used to do the mathematical operation of the paper 
    """

    d_k = query.size(-1)
    
    
    scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
    
    
    if mask is not None:
        # masked_fill_ replaces values where the condition (mask == 0) is True 
        # with a massive negative number, pushing their softmax probability to 0.
        scores = scores.masked_fill(mask == 0, -1e9)
        
    # 4. Softmax over the last dimension (the sequence length) to get probabilities
    p_attn = scores.softmax(dim=-1)
    
    # 5. Apply dropout to the attention probabilities (as described in the paper)
    if dropout is not None:
        p_attn = dropout(p_attn)
        
    # 6. Multiply the attention probabilities by V
    output = p_attn @ value
    
    # Return both the output and the attention weights (useful for visualization later)
    return output, p_attn
    


