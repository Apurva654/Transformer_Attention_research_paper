"""
 Fast local checks for the Transformer implementation. This section verifies that the core 
 architecture can run a forward pass and compute the training loss.
"""
import torch
from .model.transformer import Transformer, make_src_mask, make_tgt_mask
from .train import LabelSmoothingLoss
#--------------------------------------------------------------------------------------------------

def main():
    torch.manual_seed(7)
    vocab_size = 64
    pad_idx = 0
    model = Transformer(vocab_size=vocab_size, d_model=32, n_heads=4, 
                        n_layers=2, d_ff=128, max_seq_len=32, dropout=0.1, pad_idx=pad_idx)
    
    src = torch.tensor([
            [1, 12, 13, 14, 2, 0, 0],
            [1, 21, 22, 23, 24, 25, 2],],
            dtype=torch.long,)
    
    tgt = torch.tensor([
            [1, 31, 32, 2, 0, 0],
            [1, 41, 42, 43, 44, 2],],
        dtype=torch.long,)

    src_mask = make_src_mask(src, pad_idx)
    tgt_mask = make_tgt_mask(tgt[:, :-1], pad_idx)
    logits = model(src, tgt[:, :-1])

    criterion = LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
    loss = criterion(logits.reshape(-1, vocab_size),tgt[:, 1:].reshape(-1),)

    assert src_mask.shape == (2, 1, 1, 7)
    assert tgt_mask.shape == (2, 1, 5, 5)
    assert logits.shape == (2, 5, vocab_size)
    assert torch.isfinite(loss)

    print("Smoke test passed")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"loss: {loss.item():.4f}")
#--------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
