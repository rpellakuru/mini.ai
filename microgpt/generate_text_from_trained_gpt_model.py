import torch
from simple_gpt import GPT

# Run simple_gpt.py to re-train the model with custom hyperparameters.
# This script uses the paramters from gpt-local-trained-model-params.pth which were 
# trained by running our custome GPT implementation simple_gpt.py 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_TOKENS_TO_GENERATE = 1000
model = GPT()

model.load_state_dict(torch.load('microgpt/gpt-local-trained-model-params.pth'))
model.eval()

with open("microgpt/resources/input.txt", 'r') as f:
    content = f.read()

    # Prepare the tokens and mapping
    vocab = sorted(list(set(content)))
    vocab_size = len(vocab)

    # Simple Tokenizer for now which maps characters to integer
    stoi = {ch:i for i, ch in enumerate(vocab)}
    itos = {i:ch for i, ch in enumerate(vocab)}

    encode = lambda input: [stoi[ch] for ch in input]
    decode = lambda input: [itos[i] for i in input]


start_token = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_tokens = model.generate(start_token, MAX_TOKENS_TO_GENERATE)[0].tolist()
print(''.join(decode(generated_tokens)))

