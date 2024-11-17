import random
import numpy as np

def random_gen(start = -.5, end = .5):
    return random.uniform(start, end)


def step_function(x):
    y = int
    if x >= 0:
        y = 1
    else: #if x < 0
        y = 0
    return y



def sigmoideActivation(net):
    return 1 / (1 + np.exp(-net))

def derivativeSigmoideActivation(net):
    sig = sigmoideActivation(net)
    return sig * (1 - sig)
