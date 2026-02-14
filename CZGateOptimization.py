from optimizers.bayesian import Bayesian
from target.test import *
from process import OptimizationProcess

if __name__ == '__main__':
    target = vector_rastrigin

    dimension = 2

    bounds = [(-5, 5) for _ in range(dimension)]

    minimize = True

    optimizer = Bayesian

    process = OptimizationProcess(target, optimizer, bounds, minimize)

    process.optimize(30)


    process.solutions_pool.plot_parallel_coordinates()



