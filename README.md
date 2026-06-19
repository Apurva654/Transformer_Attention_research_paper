## Intro-
This project implements the core architecture from the research paper named- "Attention Is All You Need" in PyTorch.
The model code tries to verify the results obtained by the above paper by implmentation  of mechanisms like attention,
multi-head attention, sinusoidal positional encoding, encoder/decoder layers, label smoothing and beam-search evaluation.

//----------------------------------------------------------------------------------------------------------------------

## Repository Layout-
PAPER_NOTES.md    |    Reading notes: claim, method, datasets, metrics
README.md         |    How to install, train, and evaluate
src/              |    Transformer implementation
results/          |    Training/evaluation logs and screenshots

//----------------------------------------------------------------------------------------------------------------------
 
## Setup-
pip install -r requirements.txt

//----------------------------------------------------------------------------------------------------------------------

## Fast Architecture Check-
python -m src.smoke_test

(This command does not download data. It verifies that masks, forward pass, and
label-smoothed loss run correctly on a tiny synthetic batch.)

//----------------------------------------------------------------------------------------------------------------------

## Train-
python -m src.train --small --preset small --max-steps 12000
python -m src.train

(The paper used WMT14 EN-DE and trained for 100,000 steps on 8 P100 GPUs. 
For a practical limited-compute run, we use Multi30k.)

//----------------------------------------------------------------------------------------------------------------------

## Evaluate BLEU-
After training:
python -m src.evaluate --checkpoint checkpoints/best_model.pt --small

//----------------------------------------------------------------------------------------------------------------------

## Paper Comparison-

The paper reports Transformer Base at 27.3 BLEU on WMT14 EN-DE newstest2014.
We have tried running the same architecture, but the default submission-friendly
path uses Multi30k.

Since the paper used a larger dataset, a larger training budget, and 8 NVIDIA P100 GPU,
and we are currently working with a single GPU machine(my dear laptop), we expect a lower BLEU value
( got a low value of around 8 BLEU :/ )

Validation loss and BLEU may also select different checkpoints, so results can vary across
saved checkpoints.

//----------------------------------------------------------------------------------------------------------------------

## Important Implementation Files
- src/model/attention.py:                 scaled dot-product attention and multi-head attention.
- src/model/encoder.py:                   Transformer encoder stack.
- src/model/decoder.py:                   Transformer decoder stack.
- src/model/transformer.py:               full encoder-decoder model and masks.
- src/train.py:                           label smoothing, optimizer, learning-rate schedule, training loop.
- src/evaluate.py:                        greedy/beam decoding and SacreBLEU evaluation.
- src/data/dataset.py:                    shared BPE tokenizer and EN-DE data loaders.


-----------------------------------THANKU:)---------------------------------------------------
