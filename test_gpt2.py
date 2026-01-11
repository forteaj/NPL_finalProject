import torch
from mingpt.model import GPT
from mingpt.bpe import BPETokenizer

torch.set_grad_enabled(False)

# Load GPT-2 small
model = GPT.from_pretrained("gpt2")
model.eval()

bpe = BPETokenizer()

text = "Michelle Jones was a top-notch student. Michelle"
tokens = bpe(text)  # shape: (1, T)

# Forward pass
logits, _ = model(tokens)  # logits shape: (1, T, vocab_size)
last_logits = logits[0, -1]  # logits for next-token prediction

# Token indices (note the leading space!)
jones_idx = bpe(" Jones")[0, 0].item()
smith_idx = bpe(" Smith")[0, 0].item()

print("Text:", text)
print("Num tokens:", tokens.shape[-1])
print("Logit(' Jones'):", float(last_logits[jones_idx]))
print("Logit(' Smith'):", float(last_logits[smith_idx]))

# Show top-10 next tokens
topk = torch.topk(last_logits, k=10)
top_ids = topk.indices.tolist()
top_vals = topk.values.tolist()

print("\nTop-10 next tokens:")
for i, (tid, val) in enumerate(zip(top_ids, top_vals), 1):
    tok = bpe.decode(torch.tensor([tid]))
    print(f"{i:2d}. {tok!r:>12}  logit={val:.4f}")
