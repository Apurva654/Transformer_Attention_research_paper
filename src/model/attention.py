"""
                  Scaled Dot-Product Attention and Multi-Head Attention
                                      (Section 3.2)

  Instead of one big attention over d_model=512 dimensions, we run 8 smaller
  attention operations over d_k=64 dimensions in parallel.
   So each head can learn a different kind of relationship.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

#--------------------------------------------------------------------------------------
def scaled_dot_product_attention(
    Q: torch.Tensor,          # Queries:shape (batch, heads, seq_q, d_k)
    K: torch.Tensor,          # Keys:shape (batch, heads, seq_k, d_k)
    V: torch.Tensor,          # Values:shape (batch, heads, seq_v, d_v)
    mask: Optional[torch.Tensor] = None,  # Mask:(batch, 1, 1, seq_k)
    dropout: Optional[nn.Dropout] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    
    d_k = Q.size(-1)  # Dimension of keys/queries (64 for base model)

    # Step 1: Compute dot products between all query-key pairs
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Step 2: Apply mask BEFORE softmax
    # Masked positions get -infinity → become 0 after softmax → ignored
    if mask is not None:
    # mask=0 means "block this position"
        scores = scores.masked_fill(mask == 0, -1e9)

    # Step 3: Softmax over the key dimension
    # Converts raw scores to probabilities that sum to 1
    attention_weights = F.softmax(scores, dim=-1) 

    # Step 4: Optional dropout on attention weights (paper Section 5.4)
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # Step 5: Weighted sum of values
    output = torch.matmul(attention_weights, V)
    return output, attention_weights

#-------------------------------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0, f"d_model ({d_model}) must be divisible by h ({h})"
        self.d_model = d_model  # 512
        self.h = h              # 8 heads
        self.d_k = d_model // h # 64 dims per head

        # Four linear projections (W_Q, W_K, W_V, W_O)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)  # Query projection
        self.W_K = nn.Linear(d_model, d_model, bias=False)  # Key projection
        self.W_V = nn.Linear(d_model, d_model, bias=False)  # Value projection
        self.W_O = nn.Linear(d_model, d_model, bias=False)  # Output projection
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None  # Stored for visualization

#-------------------------------------------------------------------------------------------------
    # Split the last dimension into (h, d_k) and transpose for attention.
    #This allows attention to be computed per head in parallel.
       
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        # Reshape: (batch, seq, d_model) → (batch, seq, h, d_k)

        x = x.view(batch_size, seq_len, self.h, self.d_k)

        # Transpose: (batch, seq, h, d_k) → (batch, h, seq, d_k)
        return x.transpose(1, 2)
    
#-------------------------------------------------------------------------------------------------
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
              mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        This code snippet contatins the Forward pass
        For SELF-attention (encoder, decoder masked): Q=K=V=x
        For CROSS-attention (decoder cross-attn): Q=decoder=x, K=V=encoder_output
        """
        
        batch_size = Q.size(0)

        # Step 1: Linear projections and split into h heads
        # Each: (batch, seq, d_model) → (batch, h, seq, d_k)
        Q = self.split_heads(self.W_Q(Q))  # (batch, 8, seq_q, 64)
        K = self.split_heads(self.W_K(K))  # (batch, 8, seq_k, 64)
        V = self.split_heads(self.W_V(V))  # (batch, 8, seq_v, 64)

        # Step 2: Apply attention to all heads in parallel
        # attn_output: (batch, 8, seq_q, 64)
        attn_output, self.attention_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout)
        
        # Step 3: Concatenate heads
        # (batch, 8, seq_q, 64) → (batch, seq_q, 8, 64) → (batch, seq_q, 512)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.h * self.d_k)

        # Step 4: Final output projection W_O
        # (batch, seq_q, 512) → (batch, seq_q, 512)
        return self.W_O(attn_output)