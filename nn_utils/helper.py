__all__ = []
from typing import Any
import torch


g = torch.Generator().manual_seed(2147483647)


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
        xhat = (x - xmean)/torch.sqrt(xvar + self.eps) # Normalize to Unit variance
        self.out = self.gamma * xhat + self.beta

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





    
l1 = Linear(6, 20)
x1 = torch.randn(32, 3, 2)
linear_Val = l1(x1.view(-1, 6))
print(linear_Val)
print(linear_Val.shape)

