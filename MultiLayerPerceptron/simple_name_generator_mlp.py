# Learnings and practice from Andrey Karpathy teachings
import torch
import random
import torch.nn.functional as  F
import matplotlib.pyplot as plt

class NameGenerator:
    BLOCK_SIZE = 3
    MINI_BATCH_SIZE = 32
    HIDDEN_LAYER_SIZE = 200

    def __init__(self, words, embedding_size) -> None:
        self.g = torch.Generator().manual_seed(2147483647)
        random.seed(42)
        self.words = words
        self.embedding_size = embedding_size
        self.initialize()
        

    def initialize(self):
        random.shuffle(self.words)
        chars = sorted(set(''.join(self.words)))
        self.stoi = {s: i + 1 for i, s in enumerate(chars)}
        self.stoi['.'] = 0
        self.itos = { i:s for s, i in self.stoi.items()}
        print(f"Total words in the dataset: {len(self.words)}. Total unique characters/tokens: {len(self.stoi)}")
        self.vocab_size = len(self.stoi)

    def get_training_dataset(self):
        return self.build_dataset(self.words[: int(0.8 * len(self.words))])
    
    def get_validation_dataset(self):
        return self.build_dataset(self.words[int(0.8 * len(self.words)): int(0.9 * len(words))])
    
    def get_test_dataset(self):
        return self.build_dataset(self.words[int(0.9 * len(self.words)):])


    def mlp(self, X, Y):
        # Generate Embeddings for the input tokens. There are mathematical representation of a unit.
        # After training, the vectors will be adjusted in such a way that they make some sense
        self.C  = self.set_embeddings(self.vocab_size)
        self.W1 = torch.randn(self.BLOCK_SIZE * self.embedding_size, self.HIDDEN_LAYER_SIZE, generator=self.g)
        self.b1 = torch.randn(self.HIDDEN_LAYER_SIZE, generator=self.g)
        self.W2 = torch.randn(self.HIDDEN_LAYER_SIZE, self.vocab_size, generator=self.g)
        self.b2 = torch.randn(self.vocab_size, generator=self.g)

        parameters = [self.C, self.W1, self.b1, self.W2, self.b2]
        for p in parameters:
            p.requires_grad = True
        lossi = []

        steps = 200_000
        for i in range(steps):
            ix = torch.randint(0, X.shape[0], (self.MINI_BATCH_SIZE, ), generator=self.g)
            input_embeddings = self.C[X[ix]]
            emb_vectors_concat = input_embeddings.view(input_embeddings.shape[0], -1)
            hpreact = emb_vectors_concat @ self.W1 + self.b1
            h = torch.tanh(hpreact)
            logits = h @ self.W2 + self.b2
            loss = F.cross_entropy(logits, Y[ix])

            for p in parameters:
                p.grad = None
            
            #backward pass
            loss.backward()

            # update gradients
            lr = 0.1 if i < 100_000 else 0.01
            for p in parameters:
                p.data += -lr * p.grad
            if i % 10_000 == 0:
                print(f"{i:7d}/ {steps:7d}/ {loss:.4f}")
            lossi.append(loss.item())
        # print(loss)
        # plt.plot(lossi)
        # plt.show()


    def set_embeddings(self, vocab_size):
         self.C  = torch.randn(vocab_size, self.embedding_size, generator=self.g)

    def get_embeddings(self):
        return self.C

    def build_dataset(self, words):
        X, Y = [], []
        for word in words:
            context = [0] * self.BLOCK_SIZE
            for ch in word + '.':
                iy =  self.stoi[ch]
                X.append(context)
                Y.append(iy)
                context = context[1:] + [iy]
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        print(X.shape, Y.shape)
        return X, Y
    
    @torch.no_grad
    def loss(self, X, Y):
        emb = self.C[X]
        embcat = emb.view(emb.shape[0], -1)
        h = torch.tanh(embcat @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        loss = F.cross_entropy(logits, Y)
        print(loss.item())
        return loss.item()
    

    def generate_names(self, n_names):
        g = torch.Generator().manual_seed(2147483647 + 10)
        for _ in range(n_names):
            output = []
            context = [0] * self.BLOCK_SIZE
            while True:
                emb = self.C[torch.tensor([context])]
                h = torch.tanh(emb.view(1, -1) @ self.W1 + self.b1)
                logits = h @ self.W2 + self.b2
                probs = F.softmax(logits, dim=1)
                ix = torch.multinomial(probs, num_samples=1, generator=g).item()
                output.append(ix)
                context = context[1:] + [ix]
                if ix == 0:
                    break
            print(''.join(self.itos[i] for i in output))





if __name__ == "__main__":
    # words = open('MultiLayerPerceptron/indian-names.txt', 'r').read().splitlines()
    words = open('MultiLayerPerceptron/names.txt', 'r').read().splitlines()
    ng = NameGenerator(words, 10)
    X, Y = ng.get_training_dataset()
    ng.mlp(X, Y)

    Xv, Yv = ng.get_validation_dataset()
    ng.loss(Xv, Yv)

    print("Generating names based on the model trained")
    ng.generate_names(20)





# def mlp(X, Y, vocab_size):
#     g = torch.Generator().manual_seed(2147483647)

#     # Generate Embeddings for the input tokens. There are mathematical representation of a unit.
#     # After training, the vectors will be adjusted in such a way that they make some sense
#     C  = torch.randn(vocab_size, EMBEDDING_DIMENSION, generator=g)
#     W1 = torch.randn(X.shape[1] * EMBEDDING_DIMENSION, HIDDEN_LAYER_SIZE, generator=g)
#     b1 = torch.randn(HIDDEN_LAYER_SIZE, generator=g)
#     W2 = torch.randn(HIDDEN_LAYER_SIZE, vocab_size, generator=g)
#     b2 = torch.randn(vocab_size, generator=g)

#     parameters = [C, W1, b1, W2, b2]
#     for p in parameters:
#         p.requires_grad = True

    
#     # Use tanh function for neuron activation for hidden layers. [Other functions to explore, Relu, Sigmoid etc. See how it affects NN
#     # forward pass
    
#     for _ in range(1000):
#         input_embeddings = C[X]
#         h = torch.tanh(input_embeddings.view(-1, X.shape[1] * EMBEDDING_DIMENSION) @ W1 + b1)
#         logits = h @ W2 + b2
#         loss = F.cross_entropy(logits, Y)

#         for p in parameters:
#             p.grad = None
#         #backward pass
#         loss.backward()
#         print(loss)
#         # update gradients
#         for p in parameters:
#             p.data += -0.1 * p.grad

#     print(loss)


# def mlp_minibatch(X, Y, vocab_size):
#     # Generate Embeddings for the input tokens. There are mathematical representation of a unit.
#     # After training, the vectors will be adjusted in such a way that they make some sense
#     C  = torch.randn(vocab_size, EMBEDDING_DIMENSION, generator=g)
#     W1 = torch.randn(X.shape[1] * EMBEDDING_DIMENSION, HIDDEN_LAYER_SIZE, generator=g)
#     b1 = torch.randn(HIDDEN_LAYER_SIZE, generator=g)
#     W2 = torch.randn(HIDDEN_LAYER_SIZE, vocab_size, generator=g)
#     b2 = torch.randn(vocab_size, generator=g)

#     parameters = [C, W1, b1, W2, b2]
#     for p in parameters:
#         p.requires_grad = True

    
#     # Use tanh function for neuron activation for hidden layers. [Other functions to explore, Relu, Sigmoid etc. See how it affects NN
#     # forward pass
#     lossi = []
#     steps = 200_001
#     for i in range(steps):
#         ix = torch.randint(0, X.shape[0], (MINI_BATCH_SIZE, ), generator=g)
#         input_embeddings = C[X[ix]]
#         emb_vectors_concat = input_embeddings.view(input_embeddings.shape[0], -1)
#         hpreact = emb_vectors_concat @W1 + b1
#         h = torch.tanh(hpreact)
#         logits = h @ W2 + b2
#         loss = F.cross_entropy(logits, Y[ix])

#         for p in parameters:
#             p.grad = None
#         #backward pass
#         loss.backward()
#         # update gradients
#         lr = 0.1 if i < 100_000 else 0.01
#         for p in parameters:
#             p.data += -lr * p.grad
#         if i % 10_000 == 0:
#             print(f"{i:7d}/ {steps:7d}/ {loss:.4f}")
#         lossi.append(loss.item())

#     print(loss)
#     plt.plot(lossi)
#     plt.show()







# if __name__ == "__main__":
#     words = open('MultiLayerPerceptron/names.txt', 'r').read().splitlines()
#     ng = NameGenerator(words)
#     X, Y = ng.get_training_dataset()
#     ng.mlp(X, Y)


    # chars = sorted(set(''.join(words)))
    # stoi = {s: i + 1 for i, s in enumerate(chars)}
    # stoi['.'] = 0
    # itos = { i:s for s, i in stoi.items()}
    # print(f"Total words in the dataset: {len(words)}. Total unique characters/tokens: {len(stoi)}")
    # print(stoi)
    # n1 = int(0.8 * len(words))
    # n2 = int(0.9 * len(words))
    # random.shuffle(words)
    # Xtr, Ytr = build_dataset(words[:n1], stoi)
    # Xdev, Ydev = build_dataset(words[n1:n2], stoi)
    # Xtst, Ytst = build_dataset(words[n2:], stoi)
    # mlp_minibatch(Xtr, Ytr, len(stoi))
    # main()