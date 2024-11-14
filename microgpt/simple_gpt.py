'''
    The Bigram model is doing what it is supposed to do, but, the tokens generated seems a bit random as
    the decoder is using only the prev char to predict the next. Now let's try to code gpt like text generation
    using transformers architecture that is explained in the ***All you Need Attention*** paper. This may need GPU 
    to train, but, will try to reduce the number of params, so that, it can run on a CPU (Fingers crossed)
'''

