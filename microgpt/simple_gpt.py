import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    The Bigram model is doing what it is supposed to do, but, the tokens generated seems a bit random as
    the decoder is using only the prev char to predict the next. Now let's try to code gpt like text generation
    using transformers architecture that is explained in the ***All you Need Attention*** paper. This may need GPU 
    to train, but, will try to reduce the number of params, so that, it can run on a CPU (Fingers crossed)
'''


batch_size = 32 # Number of samples/input sequences we process in parallel
block_size = 8 # We can also call it as context size
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_eval_iter = 200
loss_eval_interval = 300
torch.manual_seed(1337)
max_iters=5000
embedding_dim = 32
layers = 3
num_heads = 4
dropout = 0



# Boilerplate code that we see in other files ------- START -----------
# Get the complete workds
with open("microgpt/resources/input.txt", 'r') as f:
    content = f.read()

# Prepare the tokens and mapping
vocab = sorted(list(set(content)))
vocab_size = len(vocab)
print("Vocab Size", vocab_size)

print(f"Vocab length: {len(vocab)}")

# Simple Tokenizer for now which maps characters to integer
stoi = {ch:i for i, ch in enumerate(vocab)}
itos = {i:ch for i, ch in enumerate(vocab)}

encode = lambda input: [stoi[ch] for ch in input]
decode = lambda input: [itos[i] for i in input]

# Prepare training dataset and testing dataset
# 90% should be allocated for traning and use the remaining 10% for testing the model
data = torch.tensor(encode(content))
training_data = data[:int(len(content)*0.9)]
testing_data = data[int(len(content)*0.9):]

x = training_data[:block_size]
y = training_data[1:block_size + 1]

# for i in range(block_size):
#     print(f"{x[:i + 1]} predicts {y[i]}")

print(data)


def get_batch(split):
    data = training_data if split == "train" else testing_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y


# for b in range(batch_size):
#     for t in range(block_size):
#          context = xb[b, :t + 1]
#          target = yb[b, t]
#          # print(f"{context.tolist()} predicts {target}")



# boilerplate code ends ---------- END --------------

@torch.no_grad
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(loss_eval_iter)
        for i in range(loss_eval_iter):
            x, y = get_batch(split)
            logits, loss = m(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out



class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.project_to_residual_path = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.project_to_residual_path(out)
        out = self.dropout(out)
        return out


class Head(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):       
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class FeedForward(nn.Module):
    def __init__(self, embedding_dim) -> None:
        super().__init__()
        self.learn_after_attention = nn.Sequential(nn.Linear(embedding_dim, 4 * embedding_dim), 
                                                   nn.ReLU(), nn.Linear(4 * embedding_dim, embedding_dim),
                                                   nn.Dropout(dropout))

    def forward(self, x):
        return self.learn_after_attention(x)








class Block(nn.Module):
    def __init__(self, embedding_dim, num_heads) -> None:
        super().__init__()
        head_size = embedding_dim//num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ff = FeedForward(embedding_dim)
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)
    

    def forward(self, x):
        x = x + self.sa(self.layer_norm_1(x))
        x = x + self.ff(self.layer_norm_2(x))
        return x



class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.blocks = nn.Sequential(*[Block(embedding_dim, num_heads=4) for _ in range(layers)])
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

        # In Attention paper this is cosine and sine function, but, GPT is using as another learning parameter
        self.position_embedding_table = nn.Embedding(block_size, embedding_dim) 


    def forward(self, idx, target=None):
        B, T = idx.shape
        token_embeddings = self.tokens_embedding_table(idx) # -> B, T, embedding_dim
        position_embeddings = self.position_embedding_table(torch.arange(T, device=device)) # -> T, embedding_dim
        x = token_embeddings + position_embeddings
        x = self.blocks(x) 
        x = self.layer_norm(x)
        logits = self.lm_head(x) # -> B, T, vocab_size
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
    
    # Generate tokens using the Multi Head Attention
    def generate(self, idx, max_tokens_to_generate):
        for _ in range(max_tokens_to_generate):
            idx_context = idx[:,-block_size:]
            logits, loss = self(idx_context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idxn = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idxn), dim=1)
        return idx
    






m = GPT()
m.to(device)
xb, yb = get_batch("train")
print(f"Traning xb shape {xb.shape}")
print(xb)

print(f"Traning yb shape {yb.shape}")
print(yb)
logits, loss = m(xb, yb)

print(f"Logits Shape: {logits.shape}, Loss: {loss}")

start_token = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_tokens = m.generate(start_token, 100)[0].tolist()
print(''.join(decode(generated_tokens)))



optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)

for step in range(max_iters):
    if step % loss_eval_interval == 0:
        losses = estimate_loss()
        print(f"Iteration {step}, Training Loss: {losses['train']:.4f}, Valuation Loss: {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


print("\nText After Training")

generated_tokens = m.generate(start_token, 100)[0].tolist()
print(''.join(decode(generated_tokens)))









    
