import random

def random_gen(start = -.5, end = .5):
    return random.uniform(start, end)


def step_function(x):
    y = int
    if x >= 0:
        y = 1
    else: #if x < 0
        y = 0
    return y