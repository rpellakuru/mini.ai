import torch
from typing import Any
from simple_name_generator_mlp import NameGenerator
import torch.nn.functional as F
from matplotlib import pyplot as plt

g = torch.Generator().manual_seed(2147483647)
EMB_DIM = 10
CONTEXT_BLOCK_SIZE = 3
N_HIDDEN = 100
VOCAB_SIZE = 87
MAX_TRAINING_STEPS = 200_000
TRAINING_BATCH_SIZE = 32

'''
    Pytorchify the code, so that, it alighns with torch.nn module
    
    Neural Networks doesn't work magically. You will need to understand its internals to build an effective system.

    Undestanding different neural network layers with raw implementations rather than using the torch.nn modules directly
    Goal: Get the intuition how things work behind the scene. 
    Ex: how the parameters behave with different strategies, Kaimin initialization, batch initialization etc
'''
class Linear:
    def __init__(self, fan_in, fan_out, bias=True) -> None:
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in ** 0.5
        self.bias = torch.zeros(fan_out) if bias else None # Why zeroes? Shouldn't it be a random values? May be this is for the NN where normalization techiniques are used

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
    
class BatchNorm1D:
    def __init__(self, dim, eps=1e-5, momentum=0.1) -> None:
        self.eps = eps
        self.dim = dim
        self.training = True
        self.momentum = momentum # Running mean/std momentum

        # Batch Normalization Parameters trained during back propogration
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        #Buffers (Trained during running momentum updated. (NO Backpropogation))
        # Make sure torch knows that it doesn't need to build the computation graph for Gradient descent for the following.
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)


    def __call__(self, x) -> Any:
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        # Batch normalization formula
        xhat = (x - xmean)/torch.sqrt(xvar + self.eps) # Normalize to Unit variance. 
        self.out = self.gamma * xhat + self.beta # More details in the batch normalization paper -> https://arxiv.org/abs/1502.03167

        # update running mean and variance
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __init__(self) -> None:
        pass


    def __call__(self, x) -> Any:
        self.out = torch.tanh(x)
        return self.out
    

    def parameters(self):
        return []



class Embedding:
    def __init__(self, num_embeddings, embedding_dim) -> None:
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out
    
    def parameters(self):
        return [self.weight]
    

class Flatten:
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out
    
    def parameters(self):
        return []
    

class Sequential:
    def __init__(self, layers) -> None:
        self.layers = layers

    def __call__(self, x) -> Any:
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]



model = Sequential([
    Embedding(VOCAB_SIZE, EMB_DIM),
    Flatten(),
    Linear(EMB_DIM * CONTEXT_BLOCK_SIZE, N_HIDDEN), BatchNorm1D(N_HIDDEN), Tanh(),
    Linear(N_HIDDEN, VOCAB_SIZE)
])

parameters = model.parameters()

print(sum(p.nelement() for p in parameters))

for p in parameters:
    p.requires_grad = True


words = open('MultiLayerPerceptron/resources/indian-names.txt', 'r').read().splitlines()
ng = NameGenerator(words, EMB_DIM)
X, Y = ng.get_training_dataset()


lossi = []


for i in range(MAX_TRAINING_STEPS):
    # mini batch
    ix = torch.randint(0, X.shape[0], (TRAINING_BATCH_SIZE, ), generator=g)

    logits = model(X[ix])
    loss = F.cross_entropy(logits, Y[ix])
               
    for p in parameters:
        p.grad = None
    
    loss.backward()
    lr = 0.1 if i < 150_000 else 0.01

    for p in parameters:
        p.data += -lr * p.grad

    if i % 10_000 == 0:
        print(f"{i:7d}/ {MAX_TRAINING_STEPS:7d}/ {loss.item():.4f}")
    lossi.append(loss.log10().item())
    #visualize_histogram()
    #break
   
    
plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))
plt.show()


@torch.no_grad
def loss(type, X, Y):
    logits = model(X)
    loss = F.cross_entropy(logits, Y)
    print(type, loss.item())
    return loss.item()




Xv, Yv = ng.get_validation_dataset()
loss("Validation", Xv, Yv)


# Evaluate
for layer in model.layers:
    layer.training = False


for _ in range(20):
    out = []
    context = [0] * CONTEXT_BLOCK_SIZE
    while True:
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(ng.itos[i] for i in out))





# print("Generating names based on the model trained")
# ng.generate_names(20)
    
# l1 = Linear(6, 20)
# x1 = torch.randn(32, 3, 2)
# linear_Val = l1(x1.view(-1, 6))
# print(linear_Val)
# print(linear_Val.shape)

