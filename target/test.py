import numpy as np

from operator import mul
from functools import reduce


def vector_sum(vector):
    vector = np.array(vector)
    return np.sum(vector)


def vector_quadratic_sum(vector):
    vector = np.array(vector)
    return np.sum(vector * vector)


def vector_mul(vector):
    vector = np.array(vector)
    return reduce(mul, vector)


def vector_trig(vector):
    vector = np.array(vector)

    vector = np.sin(vector + np.cos(vector)) * np.cos(vector) ** 2 + vector + vector * vector - np.sin(vector)*vector
    return np.sum(vector)

def vector_rastrigin(vector):
    vector = np.array(vector)

    return 10 * len(vector) + np.sum(vector * vector - 10 * np.cos(2 * np.pi * vector))
