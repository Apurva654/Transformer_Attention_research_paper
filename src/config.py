"""
 This file contains All hyperparameters in one place.

Every number here comes directly from the paper:
- Table 1:Model dimensions (d_model, d_ff, heads, layers)
- Table 3:Training hyperparameters (dropout, label smoothing)
- Section 5.3:Optimizer settings (Adam betas, warmup steps)
"""

class TransformerConfig:
           # MODEL ARCHITECTURE (paper Table 1 — Base model)
  
    D_MODEL: int = 512       # Dimension of all embeddings and sublayer outputs
                              # Every token representation is a 512-dim vector

    N_HEADS: int = 8         # Number of parallel attention heads
                              # Each head works on d_k=D_MODEL/N_HEADS=64 dims

    N_LAYERS: int = 6        # Number of encoder layers AND decoder layers (6 each)

    D_FF: int = 2048         # Inner dimension of feed-forward network
                              
    D_K: int = 64            # Key/Query dimension per head=D_MODEL / N_HEADS

    D_V: int = 64            # Value dimension per head=D_MODEL / N_HEADS

    DROPOUT: float = 0.1     # Dropout rate applied after each sublayer

    MAX_SEQ_LEN: int = 512   # Maximum sequence length
#-----------------------------------------------------------------------------------------
            # VOCABULARY
    VOCAB_SIZE: int = 37000  # BPE vocabulary size (shared between EN and DE)
                              # Paper uses ~37K for EN-DE, ~32K for EN-FR
    PAD_IDX: int = 0         # Index used for <PAD> token
    BOS_IDX: int = 1         # Index used for <BOS> (beginning of sequence)
    EOS_IDX: int = 2         # Index used for <EOS> (end of sequence)
    UNK_IDX: int = 3         # Index used for <UNK> (unknown token)

#-----------------------------------------------------------------------------------------
            # TRAINING (paper Section 5)

    BATCH_SIZE_TOKENS: int = 25000  # Tokens per batch
                                     # Paper uses 25K src + 25K tgt tokens per batch
    WARMUP_STEPS: int = 4000         # Steps to linearly increase LR
    ADAM_BETA1: float = 0.9
    ADAM_BETA2: float = 0.98         # Paper uses 0.98
    ADAM_EPS: float = 1e-9
    LABEL_SMOOTHING: float = 0.1     # Smoothing factor (paper Section 5.4)
    MAX_STEPS: int = 100000          # Paper trains for 100K steps
#-----------------------------------------------------------------------------------------                               
          # INFERENCE (paper Section 5.4)

    BEAM_SIZE: int = 4               # Beam search width
    LENGTH_PENALTY: float = 0.6   
#-----------------------------------------------------------------------------------------
           # PATHS

    CHECKPOINT_DIR: str = "checkpoints"
    RESULTS_DIR: str = "results"
    TOKENIZER_PATH: str = "tokenizer"
#-----------------------------------------------------------------------------------------

# Default config instance used throughout the project
cfg = TransformerConfig()