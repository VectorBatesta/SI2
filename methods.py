import random

def random_gen(start = -.5, end = .5):
    return random.uniform(start, end)


def step_function(x):
    if x >= 0:
        return 1
    else: #if x < 0
        return 0