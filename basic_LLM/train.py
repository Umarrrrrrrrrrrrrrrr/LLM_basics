import torch
import torch.nn.functional as F
from model import TinyTransformer
from utils import get_vocab, encode
from pathlib import Path

# load data
text = Path("data.txt").read_text()
stoi, itos, vocab_size = get_vocab(text)
data = torch.tensor(encode(text, stoi), dtype=torch.long)

#prepare sequences
block_size = 8
X, Y = [], []
for i in range(len(data) - block_size):
    X.append(data[i:i+block_size])
    Y.append(data[i+1:i+block_size+1])
X = torch.stack(X)
Y = torch.stack(Y)

# Model and optimizer
model = TinyTransformer(vocab_size=vocab_size, embed_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


# Training loop
for step in range(1000):
    logits = model(X)
    loss = F.cross_entropy(logits.view(-1, vocab_size), Y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"step {step}: Loss = {loss.item():.4f}")

# Save model and vocab
torch.save(model.state_dict(), "model.pth")
torch.save({'stoi': stoi, 'itos':itos}, "vocab.pth")