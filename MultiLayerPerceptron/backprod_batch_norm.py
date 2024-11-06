
# from MultiLayerPerceptron.simple_name_generator_mlp import NameGenerator

# n_embedding = 10

# words = open('MultiLayerPerceptron/resources/names.txt', 'r').read().splitlines()
# ng = NameGenerator(words, n_embedding)
# X, Y = ng.get_training_dataset()

# max_step = 200_000
# batch_size = 32
# lossi = []


# g = torch.Generator().manual_seed(2147483647)

# # Random initialize of vectors. The vector values gets updated as we train the Neural Network
# C = torch.randn((total_tokens, dim), generator=g)

# # Weights for input layer
# # W1 = torch.randn((X.shape[1] * dim, hidden_layer_perceptron), generator=g)
# W1 = torch.randn((X.shape[1] * dim, hidden_layer_perceptron), generator=g)
# b1 = torch.randn(hidden_layer_perceptron, generator=g)

# # Weight for the next layer (In this case it is the last)
# W2 = torch.randn((hidden_layer_perceptron, total_tokens), generator=g)
# b2 = torch.randn(total_tokens, generator=g)

# embeddings = C[X]

# parameters = [C, W1, b1, W2, b2]

# for p in parameters:
#     p.requires_grad = True