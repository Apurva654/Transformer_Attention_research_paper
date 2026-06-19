
import math                       
import torch
import torch.nn as nn
#----------------------------------------------------------------------------------
"""                  Paper Section 3.4
     training the model to learn token embeddings scaled by sqrt(d_model).
"""
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding= nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.d_model= d_model
        self.scale= math.sqrt(d_model) 

#-----------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * self.scale
#----------------------------------------------------------------------------------------------
    """
                      Fixed sinusoidal positional encoding 
                             (paper Section 3.5)
    
    """
class PositionalEncoding(nn.Module): 

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build PE matrix once:shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 

        # Frequency denominator:1 by 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()*(-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)   # even columns-sin
        pe[:, 1::2] = torch.cos(position * div_term)   # odd  columns-cos

        pe = pe.unsqueeze(0)                            
        self.register_buffer('pe', pe)                
#-----------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        x = x + self.pe[:, :x.size(1), :]   # slice to actual seq length
        return self.dropout(x)