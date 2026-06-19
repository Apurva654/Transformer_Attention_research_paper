
I have choosen the below research paper for my Task 3-
                 "Attention Is All You Need"
                                    Vaswani et al.,2017 (NeurIPS)

 [NOTE--The paper trained the base model for 100,000 steps on 8 NVIDIA P100 GPUs. 
 MY PROJECT is designed to make the architecture verifiable on a normal machine, with
honest logs documenting the smaller dataset and reduced training budget used.]

## 1. Central Claim
(reference taken from the intro and backgroung part of the research papaer)

 The research paper "Attention Is All You Need" claims that "Transformer Model Architecture" that relies exclusively on 
 "ATTENTION MECHANISMS" rather than on recurrence (RNNs/LSTMs i.e. Recurrent Neural Network/Long Short-Term Memory) and  
 convolution can effectively outperform all the prior sequence-to-sequence models on  machine translation.

  The authors claim that the TRANSFORMER MODEL not only achieves "superior translation quality" (as measured 
by BLEU scores) but also offers significant advantages in "parallelizability" and "training efficiency."

  It replaces the sequential token proccessing (used by the RNNs) with the concept of "SELF-ATTENTION" that 
helps it surpass the  existing state of the art models.
#------------------------------------------------------------------------------------------------------------------

## 2. Core Method Implemented /CORE ARCHITECTURE:-
(things that I need to implement in order to test the claims made.)
This project implements the Transformer encoder-decoder architecture 
from Figure 1 and Section 3 of the paper.

### ENCODER:- (6 identical layers)  
    (Section 3.1)
 # Each layer has 2 sub-layers:
  1. Multi-Head Self-Attention
  2. Position-wise Feed-Forward Network (FFN)
# Every sub-layer wrapped with:
  LayerNorm (x + Sublayer(x))

### DECODER:- (6 identical layers)  
    (Section 3.1)
  # Each layer has 3 sub-layers:
   1. Masked Multi-Head Self-Attention
   2. Multi-Head Cross-Attention over encoder output
   3. Position-wise Feed-Forward Network
# Every sub-layer wrapped with:
  LayerNorm (x + Sublayer(x))

### SCALED DOT-PRODUCT ATTENTION
    (Section 3.2.1)

### MULTI-HEAD ATTENTION
    (Section 3.2.2)

### POSITION-WISE FEED-FORWARD NETWORKS
    (Section 3.3)

### POSITIONAL ENCODING
    (Section 3.5)
#------------------------------------------------------------------------------------------------------------------

## 3. Dataset, Metric, and Baselines

             ###DATASETS- 
    (refered to section 5.2 and 6)

 ## Primary Task (English-to-German Translation)
- Dataset: WMT 2014 English-German
- Size: 4.5 million sentence pairs
- Tokenization: Byte-Pair Encoding (BPE)  
   with a shared source-target vocabulary of approximately 37,000 tokens
- Validation/Test: Standard newstest2013 (validation), newstest2014 (test)

## Secondary Task (English-to-French Translation)
- Dataset: WMT 2014 English-French
- Size: 36 million sentence pairs
- Tokenization: 32,000 word-piece tokens
- Test: newstest2014

             ###EVALUATION METRIC-
# BLEU SCORE (tokenized, newstest2014)

              ###BASELINE-
# English to German (EN-DE):
Model                 -       BLEU 
ByteNet    -                 23.75
Deep-Att + PosUnk     -      24.6 
GNMT + RL             -      24.6
ConvS2S               -      25.16
MoE (Mixture of Experts)-   26.03 
Transformer (Base)   -      27.3 
Transformer (Big)    -       28.4 

# English to  French (EN-FR): 
Model                 -       BLEU 
GNMT + RL             -       41.16
Deep-Att + PosUnk     -       39.2 
ConvS2                -       40.46 
MoE                   -       40.56 
Transformer (Big)     -       41.0 (Paper's result )
#-----------------------------------------------------------------------------------------------------------
                       OUR IMPLEMENTATION TARGET-

| Item              |      Paper                  |    MY Project             |

| Dataset           | WMT 2014 EN-DE (4.5M pairs) | Multi30k (29k pairs)      |
| Vocabulary        | 37,000 BPE tokens (shared)  | 8,000 BPE tokens          |
| Metric            | BLEU on newstest2014        | BLEU on Multi30k test set |
| Target BLEU       | 27.3 (Base model)           |       8-16                |

- Results are expected to be lower and not directly comparable when trained for
  fewer steps, with a smaller preset, on Multi30k, or on CPU/single-GPU
  hardware.


--------------------------------------THANK YOU:)---------------------------------------------


