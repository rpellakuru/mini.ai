import torch
from typing import Any
from MultiLayerPerceptron.simple_name_generator_mlp import NameGenerator
import torch.nn.functional as F
from matplotlib import pyplot as plt

g = torch.Generator().manual_seed(2147483647)

'''
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



def visualize_histogram():
    plt.figure(figsize=(20, 4))
    legends = []
    for i, layer in enumerate(layers[:-1]):
        if isinstance(layer, Tanh):
            t = layer.out
            print(f"layer {i:2d}  {layer.__class__.__name__:10s}mean {t.mean():+.2f},  std {t.std():.2f}, saturated: {(t.abs() > 0.97).float().mean() * 100:.2f}")
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'layer {i} ({layer.__class__.__name__})')
    plt.legend(legends)
    plt.show()





n_embedding = 10
block_size = 3
n_hidden = 100
vocab_size = 27

layers = [
            Linear(n_embedding * block_size, n_hidden), BatchNorm1D(n_hidden), Tanh(),
            Linear(n_hidden, n_hidden), BatchNorm1D(n_hidden), Tanh(),
            Linear(n_hidden, n_hidden), BatchNorm1D(n_hidden), Tanh(),
            Linear(n_hidden, n_hidden), BatchNorm1D(n_hidden), Tanh(),
            Linear(n_hidden, n_hidden), BatchNorm1D(n_hidden), Tanh(),
            Linear(n_hidden, vocab_size)
          ]



with torch.no_grad():
    layers[-1].weight *= 0.1
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 5/3

C = torch.randn((vocab_size, n_embedding), generator=g)
parameters = [C] + [p for layer in layers for p in layer.parameters()]
print("Total parameters being trained", sum( p.nelement() for p in parameters))

for p in parameters:
    p.requires_grad = True




words = open('MultiLayerPerceptron/resources/names.txt', 'r').read().splitlines()
ng = NameGenerator(words, n_embedding)
X, Y = ng.get_training_dataset()

max_step = 200_000
batch_size = 32
lossi = []


for i in range(max_step):
    # mini batch
    ix = torch.randint(0, X.shape[0], (batch_size, ), generator=g)
    emb = C[X[ix]]
    x = emb.view(emb.shape[0], -1)

    for layer in layers:
        x = layer(x)

    loss = F.cross_entropy(x, Y[ix])
        
    for layer in layers:
        layer.out.retain_grad()
        
    for p in parameters:
        p.grad = None
    
    loss.backward()
    lr = 0.1 if i < 100_000 else 0.01

    for p in parameters:
        p.data += -lr * p.grad

    if i % 10_000 == 0:
        print(f"{i:7d}/ {max_step:7d}/ {loss.item():.4f}")
    lossi.append(loss.log10().item())
    visualize_histogram()
    break
   
    


# Xv, Yv = ng.get_validation_dataset()
# ng.loss(Xv, Yv)

# print("Generating names based on the model trained")
# ng.generate_names(20)
    
# l1 = Linear(6, 20)
# x1 = torch.randn(32, 3, 2)
# linear_Val = l1(x1.view(-1, 6))
# print(linear_Val)
# print(linear_Val.shape)

