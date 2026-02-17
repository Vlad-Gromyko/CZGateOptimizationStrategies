from optimizers.bayesian import BayesianOptimizer
from optimizers.genetic import GeneticOptimizer
from optimizers.swarm import SwarmOptimizer
from plotters.line import LinePlotter
from target.test import *
from process import OptimizationProcess

if __name__ == '__main__':
    target = vector_rastrigin

    dimension = 20

    bounds = [(-5, 5) for _ in range(dimension)]

    minimize = True

    optimizer = GeneticOptimizer

    plotter = LinePlotter

    process = OptimizationProcess(target, optimizer, bounds, minimize, plotter)



    process.optimize(1000)

