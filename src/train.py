"""
           Training loop for the Transformer model
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from .model.transformer import Transformer
from .data.dataset import load_multi30k_data
from .config import cfg

#-----------------------------------------------------------------------------------

MODEL_PRESETS = {
    "base": {
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 6,
        "d_ff": 2048,
        "dropout": 0.1,
        "warmup_steps": 4000,},

    "small": {
        "d_model": 256,
        "n_heads": 4,
        "n_layers": 3,
        "d_ff": 1024,
        "dropout": 0.1,
        "warmup_steps": 800,},

    "tiny": {
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 512,
        "dropout": 0.1,
        "warmup_steps": 400,},
}

#--------------------------------------------------------------------------------
# LABEL SMOOTHING LOSS  (Paper Section 5.4)
""" Replaces hard one-hot targets with soft targets."""

class LabelSmoothingLoss(nn.Module):

    def __init__(self, vocab_size, padding_idx=0, smoothing=0.1):
        super().__init__()
        self.vocab_size  = vocab_size
        self.padding_idx = padding_idx
        self.smoothing   = smoothing
        self.confidence  = 1.0 - smoothing

    def forward(self, logits, target):
        # logits :log probabilities
        # target : true token indices

        smooth_dist = torch.full_like(logits, self.smoothing / (self.vocab_size - 2))
        smooth_dist[:, self.padding_idx] = 0             
        smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        # Zero out loss for padding positions
        pad_mask = (target == self.padding_idx)
        smooth_dist[pad_mask] = 0

        loss = F.kl_div(logits, smooth_dist, reduction='sum')
        n_tokens = (~pad_mask).sum().float()
        return loss / n_tokens


# -------------------------------------------------------------------------------------------
# LR SCHEDULE  (Paper Section 5.3, Equation 3)

def get_lr_lambda(d_model, warmup_steps):
   
    def lr_lambda(step):
        step = max(1, step)
        return d_model ** (-0.5) * min(step ** (-0.5),
                                        step * warmup_steps ** (-1.5))
    return lr_lambda
# -----------------------------------------------------------------------------------------
# GREEDY DECODE  (used for validation samples)

def greedy_decode(model, src, device, max_len=50):
    """
    Simple token-by-token decoding: always pick the highest probability token.
    Used only to print sample translations during training.
    """
    model.eval()
    from .model.transformer import make_src_mask, make_tgt_mask

    src = src.to(device)
    src_mask = make_src_mask(src, model.pad_idx)

    with torch.no_grad():
        enc_out = model.encode(src, src_mask)
        generated = torch.tensor([[cfg.BOS_IDX]], device=device)

        for _ in range(max_len):
            tgt_mask = make_tgt_mask(generated, model.pad_idx)
            dec_out  = model.decode(generated, enc_out, src_mask, tgt_mask)
            logits   = model.output_projection(dec_out[:, -1, :])
            next_tok = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)
            if next_tok.item() == cfg.EOS_IDX:
                break

    return generated[0, 1:].tolist()  
# -------------------------------------------------------------------------------------------

# VALIDATION
"""              Compute average loss on validation set                    """

def validate(model, val_loader, criterion, device, vocab_size):
    
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in  = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            logits  = model(src, tgt_in)
            loss = criterion(logits.contiguous().view(-1, vocab_size),tgt_out.contiguous().view(-1))
            total_loss += loss.item()
            n += 1

    return total_loss / max(n, 1)

# -------------------------------------------------------------------------------------------
# MAIN TRAINING FUNCTION

def train(
    use_small=False,
    max_steps=None,
    batch_size=32,
    max_length=150,
    preset="small",
    warmup_steps=None,):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model_cfg = dict(MODEL_PRESETS[preset])
    if warmup_steps is not None:
        model_cfg["warmup_steps"] = warmup_steps
    print(f"Model preset: {preset} | config: {model_cfg}")

# -------------------------------------------------------------------------------------------
# Data 
    train_loader, val_loader, _, tokenizer = load_multi30k_data(
        tokenizer_path=cfg.TOKENIZER_PATH,
        batch_size=batch_size,
        max_length=max_length,
        use_small=use_small
    )
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size}")

# -------------------------------------------------------------------------------------------
# Model 
    model = Transformer(
        vocab_size=vocab_size,
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        d_ff=model_cfg["d_ff"],
        max_seq_len=cfg.MAX_SEQ_LEN,
        dropout=model_cfg["dropout"],
        pad_idx=cfg.PAD_IDX
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}  (~{n_params/1e6:.1f}M)")

# -------------------------------------------------------------------------------------------
#  Loss, Optimizer, Scheduler

    criterion = LabelSmoothingLoss(vocab_size, cfg.PAD_IDX, cfg.LABEL_SMOOTHING)

    optimizer = Adam(model.parameters(), lr=1.0,
                     betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2),
                     eps=cfg.ADAM_EPS)

    scheduler = LambdaLR(optimizer,
                         lr_lambda=get_lr_lambda(model_cfg["d_model"], model_cfg["warmup_steps"]))
# -------------------------------------------------------------------------------------------
# Setup
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    log = open(f"{cfg.RESULTS_DIR}/training_log.txt", "w")
    log.write("step,epoch,loss,lr\n")

    best_val_loss = float('inf')
    global_step   = 0
    max_epochs    = 30
    step_limit     = max_steps if max_steps is not None else cfg.MAX_STEPS

    print("\n─── Training Started ───")

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for src, tgt in pbar:
            src, tgt = src.to(device), tgt.to(device)

            # Teacher forcing: feed tgt[:-1], predict tgt[1:]
            tgt_in  = tgt[:, :-1]    # input  to decoder (starts with <BOS>)
            tgt_out = tgt[:, 1:]     # target labels     (ends with <EOS>)

            optimizer.zero_grad()

            logits = model(src, tgt_in)          

            loss = criterion(
                logits.contiguous().view(-1, vocab_size),
                tgt_out.contiguous().view(-1))

            loss.backward()
            # Gradient clipping — prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss  += loss.item()
            n_batches   += 1
            current_lr   = scheduler.get_last_lr()[0]

            pbar.set_postfix(loss=f"{loss.item():.3f}",
                             lr=f"{current_lr:.6f}",
                             step=global_step)
            
            if global_step == 1 or global_step % 10 == 0:
                log.write(f"{global_step},{epoch+1},{loss.item():.4f},{current_lr:.8f}\n")
                log.flush()
# -------------------------------------------------------------------------------------------
            # Save checkpoint every 2000 steps
            if global_step % 2000 == 0:
                ckpt_path = f"{cfg.CHECKPOINT_DIR}/step_{global_step}.pt"
                torch.save({
                    'step': global_step,
                    'preset': preset,
                    'model_config': model_cfg,
                    'vocab_size': vocab_size,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, ckpt_path)
                print(f"\nCheckpoint saved → {ckpt_path}")

            if global_step >= step_limit:
                break

        avg_loss = epoch_loss / max(n_batches, 1)
        val_loss = validate(model, val_loader, criterion, device, vocab_size)
        print(f"\nEpoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")
        log.write(f"epoch_{epoch+1},summary,{avg_loss:.4f},{val_loss:.4f}\n")
        log.flush()

# -------------------------------------------------------------------------------------------
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'step': global_step,
                'preset': preset,
                'model_config': model_cfg,
                'vocab_size': vocab_size,
                'model_state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
            }, f"{cfg.CHECKPOINT_DIR}/best_model.pt")
            print(f"  ✓ Best model saved (val_loss={val_loss:.4f})")
# -------------------------------------------------------------------------------------------

        # Print a sample translation
        src_sample = next(iter(val_loader))[0][:1].to(device)
        gen = greedy_decode(model, src_sample, device)
        print(f"  Sample output tokens: {gen[:10]}...")

        if global_step >= step_limit:
            print("Max steps reached. Training complete.")
            break

    log.close()
    print(f"\nDone. Best model → {cfg.CHECKPOINT_DIR}/best_model.pt")

# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true",
                        help="Compatibility flag; this project uses Multi30k")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override training steps for limited-compute runs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=150)
    parser.add_argument("--preset", choices=MODEL_PRESETS.keys(), default="small",
                        help="Use small/tiny for CPU training, base for the paper-size model")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="Override Noam warmup; smaller values help short runs")
    args = parser.parse_args()
    train(
        use_small=args.small,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        max_length=args.max_length,
        preset=args.preset,
        warmup_steps=args.warmup_steps,
    )
