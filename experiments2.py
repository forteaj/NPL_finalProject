import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -----------------------------
# Token ids we care about
# -----------------------------
def token_id_with_space(token_str: str) -> int:
    """
    IMPORTANT: include the leading space for mid-sentence tokens, e.g. " Jones", " Smith".
    """
    return tok(token_str)[0].item()

SMITH_ID = token_id_with_space(" Smith")
JONES_ID = token_id_with_space(" Jones")

# -----------------------------
# Score (logit difference)
# -----------------------------
def score_smith_minus_jones(last_logits: torch.Tensor) -> float:
    """
    Returns: logit(" Smith") - logit(" Jones")
    Positive => model prefers " Smith" over " Jones"
    Negative => model prefers " Jones" over " Smith"
    """
    return (last_logits[SMITH_ID] - last_logits[JONES_ID]).item()

# -----------------------------
# Runs
# -----------------------------
@torch.no_grad()
def run_clean(prompt_clean: str):
    """
    Run clean prompt, save activations, return (x_clean, clean_acts, clean_last_logits, score_clean).
    clean_acts is a list of length n_layers; each element has shape (T, C).
    """
    x_clean = tok(prompt_clean).to(device)  # (1, T)
    _logits, _ = model(x_clean, save_activations=True, do_patch=False)

    clean_acts = [a.detach().clone() for a in model.saved_activations]   # list[(T,C)] length=n_layers
    clean_last = model.last_logits.detach().clone()                      # (vocab_size,)
    clean_score = score_smith_minus_jones(clean_last)

    return x_clean, clean_acts, clean_last, clean_score

@torch.no_grad()
def run_corrupted(prompt_corrupted: str):
    """
    Run corrupted prompt without patch, return (x_corr, corr_last_logits, score_corr).
    """
    x_corr = tok(prompt_corrupted).to(device)  # (1, T)
    _logits, _ = model(x_corr, save_activations=False, do_patch=False)

    corr_last = model.last_logits.detach().clone()
    corr_score = score_smith_minus_jones(corr_last)
    return x_corr, corr_last, corr_score

@torch.no_grad()
def run_corrupted_with_patch(x_corr: torch.Tensor, patch_layer: int, patch_pos: int, patch_value: torch.Tensor):
    """
    Run corrupted ids with a single activation patch.
    patch_value should be shape (C,) (i.e., clean_acts[L][P]).
    Returns (patched_last_logits, score_patched).
    """
    _logits, _ = model(
        x_corr,
        save_activations=False,
        do_patch=True,
        patch_layer=patch_layer,
        patch_pos=patch_pos,
        patch_value=patch_value
    )

    patched_last = model.last_logits.detach().clone()
    patched_score = score_smith_minus_jones(patched_last)
    return patched_last, patched_score

# -----------------------------
# MAIN: big loop over (L, P)
# -----------------------------
prompt_clean = "Michelle Jones was a top-notch student. Michelle"
prompt_corrupted = "Michelle Smith was a top-notch student. Michelle"

# 1) Clean run: cache activations
x_clean, clean_acts, clean_last, clean_score = run_clean(prompt_clean)

# 2) Corrupted baseline score
x_corr, corr_last, corr_score = run_corrupted(prompt_corrupted)

# 3) Sanity: token lengths must match for patching positions to line up
T_clean = x_clean.size(1)
T_corr = x_corr.size(1)
assert T_clean == T_corr, f"Token length mismatch: clean T={T_clean} vs corrupted T={T_corr}"

n_layers = len(clean_acts)
T = T_clean

print(f"n_layers={n_layers}, T={T}")
print(f"clean_score (Smith-Jones)    = {clean_score:.6f}")
print(f"corrupted_score (Smith-Jones)= {corr_score:.6f}")

# 4) Heatmap matrix:
#    store "rescued amount" = patched_score - corr_score
#    (positive means patch makes model more pro-Smith; negative means more pro-Jones)
diff_matrix = torch.zeros(n_layers, T, device="cpu")

# Optional: also store patched_score itself
patched_score_matrix = torch.zeros(n_layers, T, device="cpu")

for L in range(n_layers):
    for P in range(T):
        patch_value = clean_acts[L][P]                 # (C,)
        _patched_last, patched_score = run_corrupted_with_patch(x_corr, L, P, patch_value)

        diff_matrix[L, P] = patched_score - corr_score
        patched_score_matrix[L, P] = patched_score

# 5) Plot the heatmap (diff relative to corrupted baseline)
plt.figure()
plt.title("Activation patching: (patched_score - corrupted_score)\nscore = logit(' Smith') - logit(' Jones')")
plt.xlabel("Token position P")
plt.ylabel("Layer L")
plt.matshow(diff_matrix.numpy(), fignum=0)
plt.colorbar()
plt.show()

# If you want the raw patched scores instead:
# plt.figure()
# plt.title("Patched scores\nscore = logit(' Smith') - logit(' Jones')")
# plt.xlabel("Token position P")
# plt.ylabel("Layer L")
# plt.matshow(patched_score_matrix.numpy(), fignum=0)
# plt.colorbar()
# plt.show()
