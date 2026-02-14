from abc import ABC, abstractmethod
from typing import Union, List, Callable, Tuple
import plotly.graph_objects as go
import plotly.io
import matplotlib.pyplot as plt


class Solution:
    def __init__(self, vector: List[float], value: float):
        self.vector = vector
        self.value = value

    def __repr__(self):
        return f'Значение: {self.value}, Вектор : {self.vector}'


class SolutionPool:
    def __init__(self):
        self.solutions: List[Solution] = []


    def draw_history(self):
        iterations = list(range(len(self.solutions)))
        values = [solution.value for solution in self.solutions]
        plt.plot(iterations, values, '-')
        plt.show()


    def plot_parallel_coordinates(self, colorscale: str = 'sunsetdark', ):

        solutions = self.solutions

        coords_list = [sol.vector for sol in solutions]
        values = [sol.value for sol in solutions]

        dim_len = len(coords_list[0])
        if not all(len(c) == dim_len for c in coords_list):
            raise ValueError("Все векторы координат должны иметь одинаковую длину.")

        dimensions = []
        for i in range(dim_len):
            dim_values = [c[i] for c in coords_list]
            dimensions.append(
                dict(
                    values=dim_values,
                )
            )

        # Построение графика
        fig = go.Figure(
            data=go.Parcoords(
                dimensions=dimensions,
                line=dict(
                    color=values,
                    colorscale=colorscale,
                    showscale=True,
                    colorbar=dict(title="Значение функции")
                )
            )
        )

        fig.update_layout()

        fig.show()


    def add_solution(self, solution: Solution):
        self.solutions.append(solution)



    def add_solutions(self, other: SolutionPool):

        visited_vectors = [solution.vector for solution in other.solutions]

        told_solutions = []

        for item in other.solutions:
            if item.vector not in visited_vectors:
                self.add_solution(item)

                told_solutions.append(item)

        return told_solutions

    def find_min_solution(self):
        sorted_solutions = sorted(self.solutions, key=lambda x: x.value)

        return sorted_solutions[0]

    def find_max_solution(self):
        sorted_solutions = sorted(self.solutions, key=lambda x: x.value)

        return sorted_solutions[-1]


class BaseOptimizer(ABC):

    def __init__(self, callback: Callable, bounds, minimization: bool = True, *args, **kwargs):
        self.callback = callback

        self.minimization = minimization

        self.bounds = bounds

        self.solution_pool = SolutionPool()

        print(f'Оптимизатор {self.__class__.__name__} инициализирован')

    @abstractmethod
    def optimize(self, function, iterations, *args, **kwargs):
        pass
