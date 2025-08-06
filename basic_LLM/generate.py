import torch
from model import TinyTransformer
from utils import decode
import torch.nn.functional as F


# load vocab and model

vocab = torch.load("vocab.pth")
stoi, itos = vocab['stoi'], vocab['itos']
vocab_size = len(stoi)

model = TinyTransformer(vocab_size=vocab_size, embed_dim=32)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Start with the letter h
context = torch.tensor([[stoi['h']]], dtype=torch.long)

# Generate next 100 characters
for _ in range(100):
    logits = model(context)
    next_token_logits = logits[:, -1, :] #get logits for last taken
    probs = F.softmax(next_token_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    context = torch.cat([context, next_token], dim=1)

# Decode the generated sequence
print(decode(context[0].tolist(), itos))