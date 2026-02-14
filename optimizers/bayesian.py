import skopt.space
from skopt import Optimizer

import matplotlib.pyplot as plt

from typing import Optional, Callable, Type, List, Tuple

from optimizers.base import Solution, BaseOptimizer, SolutionPool


class Bayesian(BaseOptimizer):
    def __init__(self, callback: Callable,  bounds: List[Tuple[float, float]], minimization: bool = True) -> None:
        super().__init__(callback, bounds, minimization)

        bounds = [skopt.space.Real(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]

        self.oracle  = Optimizer(bounds)

    def optimize(self, function: Callable, iterations, *args, **kwargs):
        for i in range(iterations):

            ask = self.oracle.ask()

            y = function(ask)

            solution = Solution(ask, y)

            if self.minimization:
                self.oracle.tell(ask, y)
                self.callback(solution)

            else:
                self.oracle.tell(ask, -1 * y)

                self.callback(solution)


            self.solution_pool.add_solution(ask)

    def tell_solutions(self, solutions_pool: SolutionPool) -> None:
        told = self.solution_pool.add_solutions(solutions_pool)

        for item in told:
            self.oracle.tell(item[0], item[1])




