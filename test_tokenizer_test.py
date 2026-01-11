import torch
from mingpt.bpe import BPETokenizer

text = "Michelle Jones was a top-notch student. Michelle"

bpe = BPETokenizer()
tokens = bpe(text)[0]

print("Tokens:", tokens)
print("Número de tokens:", tokens.shape[-1])

tokens_str = [bpe.decode(torch.tensor([t])) for t in tokens]
print("Tokens decodificados:", tokens_str)
