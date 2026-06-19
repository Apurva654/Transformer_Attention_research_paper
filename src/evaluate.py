"""
    this file is used after training, Its job is to load a saved Transformer model 
    checkpoint and test how well it translates sentences.
"""
import os
import argparse
import importlib
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .model.transformer import Transformer, make_src_mask, make_tgt_mask
from .data.dataset import load_multi30k_data
from .config import cfg
#-----------------------------------------------------------------------------------------

#decode token ids INTO string
SPECIAL_IDS = {0, 1, 2, 3}   # <PAD>, <BOS>, <EOS>, <UNK>

#removes those special tokens before converting token IDs back into readable text.
def decode_ids(tokenizer, ids):
    clean_ids = [i for i in ids if i not in SPECIAL_IDS]
    return tokenizer.decode(clean_ids)

#-----------------------------------------------------------------------------------------
#                      BLEU COMPUTATION
"""
    this section Computes corpus BLEU with SacreBLEU v2.
    Returns the BLEU score as a float.
"""
def compute_bleu(hypotheses, references):
    try:
        bleu_module = importlib.import_module("sacrebleu.metrics")
    except ImportError as exc:
        raise ImportError(
            "sacrebleu is required for evaluation. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    BLEU = getattr(bleu_module, "BLEU")
    bleu_metric = BLEU(effective_order=True)
    result = bleu_metric.corpus_score(hypotheses, [references])
    return result.score
#----------------------------------------------------------------------------------------------
#                       BEAM SEARCH 
#                    (Paper Section 5.4)
"""
    beam_size=4, alpha=0.6 — exact paper values (Section 5.4)
"""
def beam_search(model, src, device, beam_size=4, max_len=100, alpha=0.6):
   
    model.eval()
    src      = src.to(device)
    src_mask = make_src_mask(src, model.pad_idx)

    with torch.no_grad():
        enc_out = model.encode(src, src_mask) 

        # Each beam: (sequence tensor, cumulative log score)
        beams     = [(torch.tensor([[cfg.BOS_IDX]], device=device), 0.0)]
        completed = []

        for _ in range(max_len):
            if not beams:
                break
            candidates = []

            for seq, score in beams:
                tgt_mask = make_tgt_mask(seq, model.pad_idx)
                dec_out  = model.decode(seq, enc_out, src_mask, tgt_mask)
                logits   = model.output_projection(dec_out[:, -1, :]) 
                log_probs = F.log_softmax(logits, dim=-1)

                # Expand: top beam_size next tokens
                topk_scores, topk_ids = log_probs[0].topk(beam_size)

                for tok_score, tok_id in zip(topk_scores, topk_ids):
                    new_seq   = torch.cat([seq, tok_id.view(1, 1)], dim=1)
                    new_score = score + tok_score.item()

                    if tok_id.item() == cfg.EOS_IDX:
                        length    = new_seq.size(1)
                        penalized = new_score / (length ** alpha)
                        # Store as plain list (strip BOS and EOS)
                        completed.append((new_seq[0, 1:-1].tolist(), penalized))
                    else:
                        candidates.append((new_seq, new_score))

            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_size]

            if len(completed) >= beam_size:
                break

        if not completed:
            if not beams:
                return []
            beams.sort(key=lambda x: x[1], reverse=True)
            return beams[0][0][0, 1:].tolist()

        completed.sort(key=lambda x: x[1], reverse=True)
        return completed[0][0]

#-----------------------------------------------------------------------------------------
# GREEDY DECODE  (faster, slightly lower BLEU)

def greedy_decode(model, src, device, max_len=100):
    model.eval()
    src      = src.to(device)
    src_mask = make_src_mask(src, model.pad_idx)

    with torch.no_grad():
        enc_out   = model.encode(src, src_mask)
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

#-----------------------------------------------------------------------------------------
# MAIN EVALUATION

def evaluate(checkpoint_path, use_small=False, use_greedy=False, n_examples=None, max_len=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

# Load data 
    _, _, test_loader, tokenizer = load_multi30k_data(tokenizer_path=cfg.TOKENIZER_PATH, 
                                                      batch_size=1, use_small=use_small)
    vocab_size = tokenizer.get_vocab_size()

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run training first: python -m src.train --small"
        )

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model_cfg = ckpt.get('model_config', {
        'd_model': cfg.D_MODEL,
        'n_heads': cfg.N_HEADS,
        'n_layers': cfg.N_LAYERS,
        'd_ff': cfg.D_FF,
        'dropout': cfg.DROPOUT,
    })
    print(f"Checkpoint model config: {model_cfg}")

#Load model 
    model = Transformer(
        vocab_size  = vocab_size,
        d_model     = model_cfg["d_model"],
        n_heads     = model_cfg["n_heads"],
        n_layers    = model_cfg["n_layers"],
        d_ff        = model_cfg["d_ff"],
        max_seq_len = cfg.MAX_SEQ_LEN,
        dropout     = model_cfg["dropout"],
        pad_idx     = cfg.PAD_IDX).to(device)

    model.load_state_dict(state)
    model.eval()
    print("Model loaded.\n")

#Translate
    hypotheses, references = [], []
    count = 0

    print("Generating translations...")
    for src_batch, tgt_batch in tqdm(test_loader):
        src = src_batch[:1]

        if use_greedy:
            gen_ids = greedy_decode(model, src, device, max_len=max_len)
        else:
            gen_ids = beam_search(model, src, device,
                                  beam_size=cfg.BEAM_SIZE,
                                  max_len=max_len,
                                  alpha=cfg.LENGTH_PENALTY)
            
        hyp = decode_ids(tokenizer, gen_ids)
        ref = decode_ids(tokenizer, tgt_batch[0].tolist())

        hypotheses.append(hyp)
        references.append(ref)
        count += 1

        if n_examples and count >= n_examples:
            break
#-----------------------------------------------------------------------------------------
#Sample output
    print("\n══════ Sample Translations ══════")
    for i in range(min(5, len(hypotheses))):
        print(f"  REF : {references[i]}")
        print(f"  GEN : {hypotheses[i]}")
        print()
#-----------------------------------------------------------------------------------------
# BLEU 
    score = compute_bleu(hypotheses, references)

    print("══════ BLEU Results ══════")
    print(f"  BLEU Score : {score:.2f}")
    print(f"  Paper (Transformer Base, WMT14 EN-DE) : 27.3 BLEU")
    print(f"  Expected (limited compute run)        : lower than paper; varies by checkpoint")

#-----------------------------------------------------------------------------------------
# Save
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    out_path = f"{cfg.RESULTS_DIR}/bleu_results.txt"
    with open(out_path, "w") as f:
        f.write(f"BLEU Score: {score:.2f}\n\n")
        f.write("Sample Translations:\n")
        for i in range(min(10, len(hypotheses))):
            f.write(f"REF : {references[i]}\n")
            f.write(f"GEN : {hypotheses[i]}\n\n")

    print(f"\nResults saved → {out_path}")
    print("Screenshot this output and save to results/ folder for submission.")
    return score

#-----------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--small",  action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--n",      type=int, default=None)
    parser.add_argument("--max-len", type=int, default=100)
    args = parser.parse_args()

    evaluate(args.checkpoint, args.small, args.greedy, args.n, args.max_len)
