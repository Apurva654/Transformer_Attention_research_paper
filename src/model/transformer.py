"""
                              The Full Transformer Model
      This assembles all components into the complete architecture from Figure 1 of the paper
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention
from .encoder import Encoder
from .decoder import Decoder
from .embeddings import TokenEmbedding, PositionalEncoding
#-----------------------------------------------------------------------------------------------
"""
    Create padding mask for the source sequence
    We create a binary mask where 1 implies "attend to this" and 0 implies "ignore this"
    Input:src (batch, src_len) -token indices
    Output:mask (batch, 1, 1, src_len) -shape for broadcasting over (batch, heads, q_len, k_len)
"""

def make_src_mask(src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    # (batch, src_len) → True where NOT padding
    mask=(src != pad_idx).unsqueeze(1).unsqueeze(2)
    # shape: (batch, 1, 1, src_len)
    return mask
#---------------------------------------------------------------------------------------------
"""
    Create combined padding + causal (look-ahead) mask for the target sequence
     Here Two things are masked:
    1. Padding positions (same as src mask)
    2. Future positions (causal mask — position i cannot see positions > i)
    Input:  tgt (batch, tgt_len) — token indices
    Output: mask (batch, 1, tgt_len, tgt_len) — combined mask
    """

def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    
    tgt_len = tgt.size(1)

    # Padding mask
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)

    # Causal (look-ahead) mask
    # torch.tril = lower triangular
    tgt_causal_mask = torch.tril(torch.ones((tgt_len, tgt_len),
                      device=tgt.device)).bool().unsqueeze(0).unsqueeze(0)
    
    # Combine: must satisfy BOTH conditions
    tgt_mask = tgt_pad_mask & tgt_causal_mask
    return tgt_mask
#--------------------------------------------------------------------------------------------------
"""
    The complete Transformer model for sequence-to-sequence tasks
                        Paper Table 1 (Base model):
    d_model=512, h=8 heads, N=6 layers, d_ff=2048, dropout=0.1, ~65M parameters
"""
class Transformer(nn.Module):
    
    def __init__(self, vocab_size:int, d_model:int=512, n_heads:int=8, n_layers:int=6,
        d_ff:int=2048, max_seq_len:int=512, dropout:float=0.1, pad_idx:int=0):
        super().__init__()

        self.pad_idx = pad_idx
        self.d_model = d_model

        # Shared Embeddings
        # Source and target share the same embedding matrix (paper Section 3.4)
        self.token_embedding = TokenEmbedding(vocab_size, d_model)

        # Positional Encoding 
        # Same PE module used for both encoder and decoder inputs
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        #Encoder
        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff, dropout)

        #Decoder
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, dropout)

        #Output Projection 
        # Linear layer: 
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.output_projection.weight = self.token_embedding.embedding.weight

        #Initialize Parameters
        self._init_parameters()
#-------------------------------------------------------------------------------------------

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

#-------------------------------------------------------------------------------------------
    """
        Encode the source sequence
    """
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
       
        # Embed tokens and add positional encoding
        src_emb = self.pos_encoding(self.token_embedding(src))

        # Run through encoder stack
        return self.encoder(src_emb, src_mask)
#-------------------------------------------------------------------------------------------
    """
        Decode the target sequence given encoder output
    """
    def decode( self, tgt: torch.Tensor, enc_output: torch.Tensor, src_mask: torch.Tensor,tgt_mask: torch.Tensor) -> torch.Tensor:
        tgt_emb = self.pos_encoding(self.token_embedding(tgt))
        return self.decoder(tgt_emb, enc_output, src_mask, tgt_mask)
    
#-------------------------------------------------------------------------------------------
    """
        Full forward pass (used during training)
    """
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
      
        # Build masks
        src_mask = make_src_mask(src, self.pad_idx)   
        tgt_mask = make_tgt_mask(tgt, self.pad_idx)

        # Encode source
        enc_output = self.encode(src, src_mask)     

        # Decode target
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)  

        # Project to vocabulary
        logits = self.output_projection(dec_output)   
        
        return F.log_softmax(logits, dim=-1)