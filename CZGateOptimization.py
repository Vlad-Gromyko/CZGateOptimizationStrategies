from optimizers.bayesian import BayesianOptimizer
from optimizers.genetic import GeneticOptimizer
from optimizers.swarm import SwarmOptimizer
from optimizers.gradient import AdamWL2Optimizer
from plotters.line import LinePlotter
from target.test import *
from target.gate import loss, structure_val, vector_val
from process import OptimizationProcess
import os

os.environ["JAX_PLATFORMS"] = "cuda,cpu"

if __name__ == '__main__':
    target = lambda vector: loss(vector, structure_val)
    #target = vector_rastrigin

    dimension = len(vector_val)

    bounds = [(0, 10) for _ in range(dimension)]

    minimize = True

    optimizer = AdamWL2Optimizer

    plotter = LinePlotter

    process = OptimizationProcess(target, optimizer, bounds, minimize, plotter)



    process.optimize(10000)

