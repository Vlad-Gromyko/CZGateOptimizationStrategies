from skopt.space import Real
from skopt import Optimizer

from typing import Callable, Tuple, List

from base import BaseOptimizer, Solution, SolutionPool


class BayesianOptimizer(BaseOptimizer):
    def __init__(self, target_function: Callable, bounds: List[Tuple[float, float]],
                 minimization: bool = True, *args, **kwargs):
        super().__init__(target_function, bounds, minimization, *args, **kwargs)
        self.oracle = Optimizer(self.bounds)

    def build_bounds(self, bounds: List[Tuple[float, float]]):
        return [Real(bound[0], bound[1]) for bound in bounds]

    def train(self, vector, function_value):
        if self.minimization:
            self.oracle.tell(vector, function_value)
        else:
            self.oracle.tell(vector, -1 * function_value)

    def optimize(self, rounds: int, *args, **kwargs):
        for i in range(rounds):
            ask = self.oracle.ask()

            f = self.target_function(ask)

            self.train(ask, function_value=f)

            solution = self.create_solution(ask, f)

            self.solution_pool.add_solution(solution)


