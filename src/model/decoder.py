"""
                                     Transformer Decoder
                                Paper Section 3.1 (Decoder stack)
Architecture :
             each DecoderLayer has THREE sublayers (encoder has 2):
             1. Masked Multi-Head Self-Attention
             2. Multi-Head Cross-Attention (Encoder-Decoder Attention)
             3. Position-wise Feed-Forward Network
               All wrapped with residual + LayerNorm
"""

import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward
#-------------------------------------------------------------------------------------

class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()

        # Sub-layer 1: MASKED self-attention (decoder attends to itself)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)

        # Sub-layer 2: CROSS-attention (decoder attends to encoder output)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)

        # Sub-layer 3: Feed-forward
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # Three layer norms — one per sublayer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

#---------------------------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,              # Decoder input
        enc_output: torch.Tensor,     # Encoder output
        src_mask: torch.Tensor,       # Padding mask for source
        tgt_mask: torch.Tensor        # Causal + padding mask for target
    ) -> torch.Tensor:
        
        #Sub-layer 1: Masked Self-Attention
        # Q=K=V=x (self-attention), but with causal mask
        # Mask prevents position i from seeing positions i+1, i+2, ...
        attn1_out = self.self_attn(Q=x, K=x, V=x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn1_out))

        # Sub-layer 2: Cross-Attention (Encoder-Decoder Attention) 
        # Q = decoder state
        # K = encoder output
        # V = encoder output
        # src_mask prevents attending to <PAD> in source
        attn2_out = self.cross_attn(Q=x, K=enc_output, V=enc_output, mask=src_mask)
        x = self.norm2(x + self.dropout(attn2_out))

        # Sub-layer 3: Feed-Forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))

        return x 

#----------------------------------------------------------------------------------------------- 
class Decoder(nn.Module):
    """
     This code snippet describes the Full decoder:
     stack of N=6 DecoderLayers + final LayerNorm

    """
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()

        self.layers = nn.ModuleList([ DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)])
        
        self.norm = nn.LayerNorm(d_model)
#--------------------------------------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,         #target embeddings + positional encoding
        enc_output: torch.Tensor, #final encoder output
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor) -> torch.Tensor:
        
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)