import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from mingpt.model import GPT
from mingpt.utils import set_seed
from mingpt.bpe import BPETokenizer
import torch.nn.functional as F
import matplotlib.pyplot as plt

set_seed(3407)

use_mingpt = True # use minGPT or huggingface/transformers model?
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

# Use the same BPE tokenizer as the GPT-2 model (minGPT implementation)
# This ensures a consistent mapping between text tokens and vocabulary indices
tok = BPETokenizer()



#----------
#PROMPT
#----------

prompt           = "Michelle Jones was a top-notch student. Michelle"
prompt_corrupted = "Michelle Smith was a top-notch student. Michelle"

# --- IDs ---
smith_id = tok(" Smith")[0].item()
jones_id = tok(" Jones")[0].item()

def score(last_logits):
    return (last_logits[smith_id] - last_logits[jones_id]).item()


# --- CLEAN ---
x_clean = tok(prompt).to(device)
_ , _ = model(x_clean, save_activations=True, do_patch=False)
clean_acts = [a.clone() for a in model.saved_activations]
clean_score = score(model.last_logits)

# --- CORRUPTED BASELINE ---
x_corr = tok(prompt_corrupted).to(device)
_ , _ = model(x_corr, save_activations=False, do_patch=False)
corr_score = score(model.last_logits)


# --- BIG LOOP ---
n_layers = len(clean_acts)
T = x_clean.size(1)
diff = torch.zeros(n_layers, T)

for L in range(n_layers):
    for P in range(T):
        _ , _ = model(x_corr,  save_activations=False,  do_patch=True,  patch_layer=L,  patch_pos=P,  patch_value=clean_acts[L][P])
        patched_score = score(model.last_logits)
        diff[L, P] = patched_score - corr_score


print(f"Clean score     (Smith - Jones): {clean_score:.4f}")
print(f"Corrupted score (Smith - Jones): {corr_score:.4f}")


plt.figure(figsize=(8, 6))
plt.title("Activation patching effect\n(score = logit(' Smith') - logit(' Jones'))")
plt.xlabel("Token position")
plt.ylabel("Layer")

plt.matshow(diff.numpy(), fignum=0)
plt.colorbar(label="Patched score − corrupted score")

plt.show()




#show top tokens before and after patching

def print_topk(last_logits, k=5):
    probs = F.softmax(last_logits, dim=-1)
    top_p, top_id = torch.topk(probs, k)
    for i, (p, tid) in enumerate(zip(top_p.tolist(), top_id.tolist()), 1):
        print(f"{i}. {tok.decode(torch.tensor([tid]))}  p={p:.3f}")

print("=== Corrupted ===")
_ , _ = model(x_corr, save_activations=False, do_patch=False)
print_topk(model.last_logits)

print("\n=== Patched (L=4, P=1) ===")
_ , _ = model(x_corr, save_activations=False, do_patch=True,
              patch_layer=4, patch_pos=1, patch_value=clean_acts[4][1])
print_topk(model.last_logits)













