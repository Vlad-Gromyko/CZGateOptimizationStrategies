from optimizers.bayesian import BayesianOptimizer
from optimizers.genetic import GeneticOptimizer
from optimizers.swarm import SwarmOptimizer
from plotters.line import LinePlotter
from target.test import *
from target.gate import loss, structure_val, vector_val
from process import OptimizationProcess


if __name__ == '__main__':
    target = lambda vector: loss(vector, structure_val)
    #target = vector_rastrigin

    dimension = len(vector_val)

    bounds = [(-0, 7) for _ in range(dimension)]

    minimize = True

    optimizer = BayesianOptimizer

    plotter = LinePlotter

    process = OptimizationProcess(target, optimizer, bounds, minimize, plotter)



    process.optimize(1000)

