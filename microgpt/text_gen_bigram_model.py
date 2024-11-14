
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    A simple Bigram model which just looks at the prev character and then predicts the next token.
'''
# Get the complete workds
with open("microgpt/resources/input.txt", 'r') as f:
    content = f.read()

# Prepare the tokens and mapping
vocab = sorted(list(set(content)))

print(f"Vocab length: {len(vocab)}")

# Simple Tokenizer for now which maps characters to integer
stoi = {ch:i for i, ch in enumerate(vocab)}
itos = {i:ch for i, ch in enumerate(vocab)}

encode = lambda input: [stoi[ch] for ch in input]
decode = lambda input: [itos[i] for i in input]

print(encode("Hi! Hello Everyone"))
print(''.join(decode(encode("Hi! Hello Everyone"))))

# Prepare training dataset and testing dataset
# 90% should be allocated for traning and use the remaining 10% for testing the model



block_size = 8
data = torch.tensor(encode(content))
training_data = data[:int(len(content)*0.9)]
testing_data = data[int(len(content)*0.9):]

x = training_data[:block_size]
y = training_data[1:block_size + 1]

# for i in range(block_size):
#     print(f"{x[:i + 1]} predicts {y[i]}")

batch_size = 4
# Set a seed for random generation
torch.manual_seed(1337)
print(data)


def get_batch(split):
    data = training_data if split == "train" else testing_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    return x,y

xb, yb = get_batch("train")
print(f"Inputs xb shape {xb.shape}")
print(xb)

print(f"Inputs yb shape {yb.shape}")
print(yb)



for b in range(batch_size):
    for t in range(block_size):
         context = xb[b, :t + 1]
         target = yb[b, t]
         # print(f"{context.tolist()} predicts {target}")

vocab_size = len(vocab)
print("Vocab Size", vocab_size)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tokens_embedding = nn.Embedding(vocab, vocab)

    def forward(self, idx, target=None):
        logits = self.tokens_embedding(idx)
        # print(f"Initial logits shape: {logits.shape}")
        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            # Pytorch expects the dimentsions ins B, C, T. But, to stick to
            # what we learned so far, let's squeeze this to a 2-D that we worked so far
            logits = logits.view(B * T, C)
            loss = F.cross_entropy(logits, target.view(B * T))
        return logits, loss
    
    # Generate tokens using the simple bigram model.
    def generate(self, idx, max_tokens_to_generate):
        for _ in range(max_tokens_to_generate):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idxn = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idxn), dim=1)
        return idx
    

m = BigramLanguageModel(vocab_size)

logits, loss = m(xb, yb)

print(f"Logits Shape: {logits.shape}, Loss: {loss}")

start_token = torch.zeros((1, 1), dtype=torch.long)
generated_tokens = m.generate(start_token, 100)[0].tolist()
print(''.join(decode(generated_tokens)))



optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)

for steps in range(100_000):
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


print("\nText After Training")

generated_tokens = m.generate(start_token, 100)[0].tolist()
print(''.join(decode(generated_tokens)))









    
