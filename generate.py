import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from mingpt.model import GPT
from mingpt.utils import set_seed
from mingpt.bpe import BPETokenizer
set_seed(3407)

use_mingpt = True # use minGPT or huggingface/transformers model?
#we are supposed to be using gpt2, so I changed it
#model_type = 'gpt2-xl'
model_type = 'gpt2'

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

if use_mingpt:
    model = GPT.from_pretrained(model_type)
else:
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.config.pad_token_id = model.config.eos_token_id # suppress a warning

# ship model to device and set to eval mode
model.to(device)
model.eval();


def generate(prompt='', num_samples=10, steps=20, do_sample=True):
    # tokenize the input prompt into integer input sequence
    if use_mingpt:
        tokenizer = BPETokenizer()
        if prompt == '':
            # to create unconditional samples...
            # manually create a tensor with only the special <|endoftext|> token
            # similar to what openai's code does here https://github.com/openai/gpt-2/blob/master/src/generate_unconditional_samples.py
            x = torch.tensor([[tokenizer.encoder.encoder['<|endoftext|>']]], dtype=torch.long)
        else:
            x = tokenizer(prompt).to(device)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(model_type)
        if prompt == '':
            # to create unconditional samples...
            # huggingface/transformers tokenizer special cases these strings
            prompt = '<|endoftext|>'
        encoded_input = tokenizer(prompt, return_tensors='pt').to(device)
        x = encoded_input['input_ids']

    # we'll process all desired num_samples in a batch, so expand out the batch dim
    x = x.expand(num_samples, -1)

    # forward the model `steps` times to get samples, in a batch
    y = model.generate(x, max_new_tokens=steps, do_sample=do_sample, top_k=40)

    for i in range(num_samples):
        out = tokenizer.decode(y[i].cpu().squeeze())
        print('-' * 80)
        print(out)


#generate(prompt='Andrej Karpathy, the', num_samples=10, steps=20)

#new

import torch.nn.functional as F


# Use the same BPE tokenizer as the GPT-2 model (minGPT implementation)
# This ensures a consistent mapping between text tokens and vocabulary indices
tok = BPETokenizer()

@torch.no_grad()
def get_last_logits(prompt: str):
    """
       Runs a forward pass of the model on the given prompt and
       returns the logits corresponding to the prediction of the next token.

       Parameters:
           prompt (str): Input text fed to the model.

       Returns:
           last_logits (Tensor): Logits for the next-token prediction (vocab_size,).
           input_ids (Tensor): Tokenized input sequence (T,).
       """
    # Tokenize the prompt and move it to the same device as the model
    x = tok(prompt).to(device)     # (1, T)

    # Forward pass through the model
    # logits has shape (1, T, vocab_size)
    logits, _ = model(x)                 # (1, T, vocab)

    # Extract logits for the last token position
    # These logits correspond to the prediction of the next token
    return logits[0, -1], x[0]  # last_logits, tokenizer, input_ids_1d

def topk_next_tokens(last_logits, tokenizer, k=20):
    """
        Displays the top-k most likely next tokens according to the model.

        Parameters:
            last_logits (Tensor): Logits for the next-token prediction.
            tokenizer: Tokenizer used to decode token ids.
            k (int): Number of top tokens to display.
        """
    # Convert logits to probabilities
    probs = F.softmax(last_logits, dim=-1)

    # Select the k tokens with highest probability
    top_probs, top_ids = torch.topk(probs, k)

    # Decode and print the top-k predictions
    for i, (p, tid) in enumerate(zip(top_probs.tolist(), top_ids.tolist()), 1):
        toke = tokenizer.decode(torch.tensor([tid]))
        print(f"{i:2d}. {repr(toke):>12}  p={p:.4f}  id={tid}")


prompt_clean = "Michelle Jones was a top-notch student. Michelle"
prompt_corrupted = "Michelle Smith was a top-notch student. Michelle"

# Run the model on both inputs and extract next-token logits
last_logits, ids = get_last_logits(prompt_clean)
last_logits_corr, _ = get_last_logits(prompt_corrupted)

print("Num tokens:", len(ids))
print("Tokens:", "/".join([tok.decode(torch.tensor([int(t)])) for t in ids]))


# Display the top-k predicted next tokens for the clean input
topk_next_tokens(last_logits, tok, k=20)

smith_id = tok(" Smith")[0].item()
jones_id = tok(" Jones")[0].item()

print("Smith id:", smith_id, "Jones id:", jones_id)


# Compute the logit difference for the clean input
# A negative value means "Jones" is preferred over "Smith"
delta_clean = last_logits[smith_id].item() - last_logits[jones_id].item()

# Compute the logit difference for the corrupted input
# A positive value means "Smith" is preferred over "Jones"
delta_corrupted = last_logits_corr[smith_id].item() - last_logits_corr[jones_id].item()

print("Delta logit(Smith) - logit(Jones):", delta_clean)
print("Delta corrupted logit  (Smith) - logit(Jones):", delta_corrupted)
