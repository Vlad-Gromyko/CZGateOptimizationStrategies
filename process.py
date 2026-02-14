from typing import Optional, Callable, Type, List, Tuple
import datetime

from optimizers.base import  Solution, SolutionPool, BaseOptimizer



class OptimizationProcess:
    def __init__(self, target_function: Callable,
                 optimizer: Type[BaseOptimizer], bounds: List[Tuple[float, float]], minimization: bool = True) -> None:

        self.target_function = target_function

        self.minimization = minimization

        self.optimizer = optimizer(self.callback, bounds, minimization)


        self.solutions_pool = SolutionPool()

    def callback(self, solution: Solution):

        self.solutions_pool.add_solution(solution)

        print('-'*20)

        print(len(self.solutions_pool.solutions))

        print(f'Оптимизатор {self.optimizer.__class__.__name__} нашел значение: {solution.value}')

        print('Лучшее решение: ', self.find_best())

        print('-'*20)

    def find_best(self):
        if self.minimization:
            return self.solutions_pool.find_min_solution()
        else:
            return self.solutions_pool.find_max_solution()


    def change_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.optimizer.tell_solutions(self.solutions_pool)

    def optimize(self, iterations):
        self.optimizer.optimize(self.target_function,iterations)

    def plot(self):
        self.solutions_pool.plot_parallel_coordinates()





