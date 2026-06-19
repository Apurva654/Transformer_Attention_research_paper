"""
            Dataset and tokenizer utilities for EN-DE translation.
"""
import os
# Used for checking and creating folders/files
import torch
# Used to create PyTorch tensors from token IDs
from datasets import load_dataset
# Imports Hugging Face’s dataset loader
# This is how our code downloads/loads Multi30k
from tokenizers import ByteLevelBPETokenizer
# BPE means Byte Pair Encoding
from tokenizers.processors import TemplateProcessing
# Used to automatically add special tokens like <BOS> and <EOS> around every sentence
from torch.nn.utils.rnn import pad_sequence
# Used to pad variable-length sentences in a batch to the same length
from torch.utils.data import DataLoader, Dataset
# creates batches from the pytorch dataset class.

# these are special vocab tokens:
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
# <PAD> = padding token
# <BOS> = beginning of sentence
# <EOS> = end of sentence
# <UNK> = unknown token

#--------------------------------------------------------------------------------------------------
# this is basically a helper function.
# It takes a tokenizer and returns the same tokenizer after adding automatic sentence formatting.
def apply_post_processor(tokenizer: ByteLevelBPETokenizer) -> ByteLevelBPETokenizer:
    bos_id = tokenizer.token_to_id("<BOS>")
    eos_id = tokenizer.token_to_id("<EOS>")
    # Finds the integer IDs for <BOS> and <EOS>

    if bos_id is None or eos_id is None:
    # Checks whether those tokens exist in the tokenizer vocabulary
        raise ValueError(                           #if not then raises an error
            "Tokenizer is missing <BOS> or <EOS>." 
            " Delete tokenizer and rerun training.")

    tokenizer.post_processor = TemplateProcessing(
        single="<BOS> $A <EOS>",
        special_tokens=[("<BOS>", bos_id), ("<EOS>", eos_id)],)
    return tokenizer #Returns the updated tokenizer
#---------------------------------------------------------------------------------------------------------
"""
    this code snippet defines a function to train a shared byte-level BPE tokenizer on source and target text.
"""

def train_tokenizer(sentences, vocab_size=8000, save_path="tokenizer"):
   
    print("Training BPE tokenizer...")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator( sentences, vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL_TOKENS)
    # Trains the tokenizer using all sentences
    tokenizer = apply_post_processor(tokenizer)
    # Adds automatic <BOS> and <EOS> behavior
    os.makedirs(save_path, exist_ok=True)
    # Creates the tokenizer folder if it does not exist
    tokenizer.save_model(save_path)
    print(f"Tokenizer saved to {save_path}/")
    return tokenizer
#---------------------------------------------------------------------------------------------------------
"""
    this code snippet is used to load a saved tokenizer and restore BOS/EOS post-processing.
"""
def load_tokenizer(path="tokenizer"):
  
    vocab_file = os.path.join(path, "vocab.json")
    merges_file = os.path.join(path, "merges.txt")

    if not os.path.exists(vocab_file) or not os.path.exists(merges_file):
        raise FileNotFoundError(f"No tokenizer found at {path}. Run training first to create it.")
    #If tokenizer files are missing, raises an error

    tokenizer = ByteLevelBPETokenizer(vocab=vocab_file, merges=merges_file)
    #Loads the tokenizer from saved files
    return apply_post_processor(tokenizer)
    #Restores automatic <BOS> and <EOS> insertion and returns the tokenizer
    
#-----------------------------------------------------------------------------------------------------
"""
    this code snippet contains Tokenized translation pairs filtered by maximum sequence length.
"""
class TranslationDataset(Dataset):
   
    def __init__(self, src_sentences, tgt_sentences, tokenizer, max_length=150): #Constructor function
        #src_sentences: English sentences
       #tgt_sentences: German sentences

        self.pairs = []
        # Creates an empty list to store tokenized sentence pairs
        total = len(src_sentences)
        # Counts how many source sentences exist
        print(f"Tokenizing {total} sentence pairs...")

        for src, tgt in zip(src_sentences, tgt_sentences):
            # Loops through English and German sentences together
            if not src or not tgt or not src.strip() or not tgt.strip():
                continue 
            #Skips empty or blank sentences

            src_ids = tokenizer.encode(src).ids
            tgt_ids = tokenizer.encode(tgt).ids
            #Converts sentences into token IDs

            if len(src_ids) <= max_length and len(tgt_ids) <= max_length:
            #Only keeps sentence pairs where both English and German are not too long

                self.pairs.append(
                    (
                        torch.tensor(src_ids, dtype=torch.long), #Converts token ID lists into PyTorch tensors
                        torch.tensor(tgt_ids, dtype=torch.long),
                    )
                )

        print(f"Kept {len(self.pairs)} / {total} pairs after filtering.")
        #Prints how many sentence pairs survived filtering

    def __len__(self):
        return len(self.pairs) #Returns dataset size

    def __getitem__(self, idx):#Returns one example by index
        return self.pairs[idx]
#---------------------------------------------------------------------------------------------------------------
"""
    this code snippet defines how to combine multiple examples into one batch
"""
def collate_fn(batch, pad_idx=0):
    src_batch = [src for src, _ in batch] #Extracts all source tensors from the batch
    tgt_batch = [tgt for _, tgt in batch] #Extracts all target tensors from the batch

    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    #Pads all source sentences to the same length
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
     #Pads all target sentences to the same length
    return src_padded, tgt_padded
 #------------------------------------------------------------------------------------------------------------
"""
    this code snippet contains a helper function to read train/validation/test split.
"""
def _read_translation_split(dataset, split):
    src, tgt = [], []
    for item in dataset[split]:
        #Some Hugging Face datasets store text like this
        if "translation" in item:
            src.append(item["translation"]["en"])
            tgt.append(item["translation"]["de"])
        #Other datasets may store text directly as
        else:
            src.append(item["en"])
            tgt.append(item["de"])
    return src, tgt
#--------------------------------------------------------------------------------------------------------------
"""
    this code snippet defines a helper function for loading the smaller Multi30k dataset
"""
def _load_multi30k():

    try:
        print("Loading bentrevett/multi30k...")
        dataset = load_dataset("bentrevett/multi30k")
        #Loads Multi30k from Hugging Face

        return (
            _read_translation_split(dataset, "train"),
            _read_translation_split(dataset, "validation"),
            _read_translation_split(dataset, "test"),
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not load Multi30k. Check internet access and the installed "
            "datasets package."
        ) from exc
#----------------------------------------------------------------------------------------------------------------
""" 
        Return Multi30k train, validation, test dataloaders and the shared tokenizer.
"""
def load_multi30k_data(tokenizer_path="tokenizer", batch_size=32, max_length=150, use_small=False):
    (train_src, train_tgt), (val_src, val_tgt), (test_src, test_tgt) = _load_multi30k()
    vocab_size = 8000

    if os.path.exists(os.path.join(tokenizer_path, "vocab.json")):
        #Checks whether a tokenizer already exists
        print("Loading existing tokenizer...")
        tokenizer = load_tokenizer(tokenizer_path)
        #Loads saved tokenizer
    else:
        tokenizer = train_tokenizer(train_src + train_tgt, vocab_size, tokenizer_path)
        #Trains tokenizer on both English and German training text.

    train_ds = TranslationDataset(train_src, train_tgt, tokenizer, max_length)
    #Creates tokenized training dataset
    val_ds = TranslationDataset(val_src, val_tgt, tokenizer, max_length)
    #Creates validation dataset
    test_ds = TranslationDataset(test_src, test_tgt, tokenizer, max_length)
    #Creates test dataset

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0
    )#Creates training loader
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, collate_fn=collate_fn, num_workers=0
    )#Creates validation loader
    test_loader = DataLoader(
        test_ds, batch_size=1, collate_fn=collate_fn, num_workers=0
    )#Creates test loader

    return train_loader, val_loader, test_loader, tokenizer
#-------------------------------------------------------------------------------------------------------
load_wmt14 = load_multi30k_data
