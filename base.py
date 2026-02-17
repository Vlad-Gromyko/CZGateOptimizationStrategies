import dill
import numpy as np
import plotly.graph_objects as go
import plotly.io
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from typing import Union, List, Callable, Tuple, Optional

import datetime
import os


class Solution:
    def __init__(self, vector: List[float], value: float, function_meta_data: dict = None,
                 optimizer_meta_data: dict = None):

        self.created_at = datetime.datetime.now()

        self.vector = vector
        self.function_value = value

        self.dimension = len(self.vector)

        self.function_meta_data = function_meta_data
        self.optimizer_meta_data = optimizer_meta_data

    def __eq__(self, other):
        if isinstance(other, Solution):
            if np.array_equal(self.vector, other.vector):
                return True

        return False

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return (f'Значение: {self.function_value}'
                f'Вектор: {self.vector}'
                f'Создан: {self.created_at}\n'
                f'Оптимизатор: {self.optimizer_meta_data}'
                f'Функция: {self.function_meta_data}')


class SolutionPool:
    def __init__(self):
        self.solutions: List[Solution] = []

        self.onNewSolution: Optional[Callable | None] = None

    def add_solution(self, new_solution: Solution):

        self.solutions.append(new_solution)

        if self.onNewSolution is not None:
            self.onNewSolution(new_solution)
            return True
        else:
            return False

    def save(self, dir_path: str):
        files_num = len(os.listdir(dir_path)) + 1
        path = dir_path + '/' + str(files_num) + '.pkl'
        dill.dump(self, path)

    def load(self, path: str):
        solution_pool = dill.load(path)
        for solution in solution_pool:
            self.add_solution(solution)

    def min_solution(self):
        return self.sorted_solutions()[0]

    def max_solution(self):
        return self.sorted_solutions()[-1]

    def sorted_solutions(self, attr='function_value'):
        return sorted(self.solutions, key=lambda solution: solution.__dict__[attr])


class BaseOptimizer(ABC):
    def __init__(self, target_function: Callable, bounds: List[Tuple[float, float]], minimization: bool = True, *args,
                 **kwargs):
        super().__init__()
        self.target_function = target_function

        self.bounds = self.build_bounds(bounds)

        self.minimization = minimization

        self.solution_pool = SolutionPool()
        self.solution_pool.onNewSolution = self.tell_solution
        self.solution_listener = None

    @abstractmethod
    def build_bounds(self, bounds: List[Tuple[float, float]]):
        pass

    @abstractmethod
    def optimize(self, rounds, *args, **kwargs):
        pass

    def take_solutions(self, solution_pool: SolutionPool):
        for solution in solution_pool.solutions:
            self.solution_pool.add_solution(solution)


    def tell_solution(self, solution: Solution):
        self.solution_listener.add_solution(solution)

    def create_solution(self, vector, f, all_data: bool = False) -> Solution:
        if all_data:
            return Solution(vector, f, self.create_function__meta_data(), self.create_optimizer_meta_data())
        else:
            return Solution(vector, f)

    def create_function__meta_data(self):
        return {'name': self.target_function.__name__}

    def create_optimizer_meta_data(self):
        return {'name': self.__class__.__name__, 'minimization': self.minimization, 'bounds': self.bounds}


class BasePlotter(ABC):
    def __init__(self):
        self.fig, self.ax = plt.subplots()

        plt.ion()
        plt.show()

    @abstractmethod
    def plot_solution_pool(self, solution_pool: SolutionPool, *args, **kwargs):
        pass
