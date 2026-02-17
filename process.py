import matplotlib.pyplot as plt

from typing import Callable, Type, List, Tuple

from base import Solution, SolutionPool, BaseOptimizer, BasePlotter


class OptimizationProcess:
    def __init__(self, target_function: Callable,
                 optimizer: Type[BaseOptimizer], bounds: List[Tuple[float, float]], minimization: bool = True,
                 plotter: Type[BasePlotter] = None) -> None:

        self.target_function = target_function

        self.minimization = minimization

        optimizer = optimizer(target_function, bounds, minimization)

        self.solutions_pool = SolutionPool()

        self.optimizer = None

        self.accept_optimizer(optimizer)

        self.solutions_pool.onNewSolution = self.new_solution_callback

        if plotter:
            self.plotter = plotter()
        else:
            self.plotter = None

    def new_solution_callback(self, solution: Solution):

        if self.plotter:
            self.plotter.plot_solution_pool(self.solutions_pool)
        print('Лучшее', self.find_best().function_value)
        print(len(self.solutions_pool.solutions))

       # print('Текущее', solution)
       # print('\n\n')

    def accept_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.optimizer.solution_listener = self.solutions_pool


    def find_best(self):
        if self.minimization:
            return self.solutions_pool.min_solution()
        else:
            return self.solutions_pool.max_solution()

    def optimize(self, iterations):
        self.optimizer.optimize(iterations)
