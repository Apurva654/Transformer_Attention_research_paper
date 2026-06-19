"""
                            Transformer Encoder
       (The encoder converts the input sentence into meaningful contextual vectors)
                       Paper Section 3.1 (Encoder stack)

Architecture:  
    Single encoder layer contains two sublayers:
                                       1. Multi-Head Self-Attention
                                       2. Position-wise Feed-Forward Network
    Each sublayer uses: LayerNorm(x + Dropout(Sublayer(x)))
"""

import torch
#Used for tensor type annotations
import torch.nn as nn
# Imports PyTorch neural network modules
from .attention import MultiHeadAttention
# Imports the custom multi-head attention class
from .feed_forward import PositionwiseFeedForward
# Imports the custom feed-forward network

#-------------------------------------------------------------------------------------------
class EncoderLayer(nn.Module):
 #Defines one encoder layer, the base model uses 6.  

    # Constructor for one encoder layer.
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        #d_model:vector size, 
        # n_heads:number of attention head,
        # d_ff: hidden size inside feed-forward network
        # dropout: dropout probability (usually 0.1)

        super().__init__()
        #Calls the parent nn.Module constructor

        # Sub-layer 1: Multi-Head Self-Attention
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)

        # Sub-layer 2: Feed-Forward Network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # Layer normalization after each sublayer
        self.norm1 = nn.LayerNorm(d_model)
        # used after self-attention
        self.norm2 = nn.LayerNorm(d_model)
        # used after feed-forward

        # Dropout applied to the output of each sublayer (before adding residual)
        self.dropout = nn.Dropout(dropout)
#----------------------------------------------------------------------------------------------------
"""
    This code snippet defines how data Processing in one encoder layer is done
    The src_mask prevents attention to <PAD> tokens
"""

def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
       
        # Sub-layer 1: Self-Attention
        # Query=Key=Value=x because it's self-attention (attending to itself)
        attn_out = self.self_attn(Q=x, K=x, V=x, mask=src_mask)

        # Residual connection + LayerNorm
        x = self.norm1(x + self.dropout(attn_out))

        # Sub-layer 2: Feed-Forward
        ff_out = self.feed_forward(x)

        # Residual connection + LayerNorm
        x = self.norm2(x + self.dropout(ff_out))

        return x 
#-------------------------------------------------------------------------------------------------------
"""
    this code snippet defines the full encoder stack, 
    each composed of multiple encodelayers.
"""
class Encoder(nn.Module):
  
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float):
        # Constructor for the full encoder

        super().__init__()
        #Calls PyTorch parent constructor

        # Creates a PyTorch list of encoder layers
        self.layers = nn.ModuleList([
             #Creates n layers separate encoder layers.
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        # Final layer normalization 
        self.norm = nn.LayerNorm(d_model)
#-----------------------------------------------------------------------------------------------------------

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:

       #Loops through all encoder layers
        for layer in self.layers:
            x = layer(x, src_mask)

        # Final layer normalization
        return self.norm(x)