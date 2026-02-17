import matplotlib.pyplot as plt

from base import SolutionPool, Solution, BasePlotter


class LinePlotter(BasePlotter):
    def plot_solution_pool(self, solution_pool: SolutionPool, *args, **kwargs):
        min_history, current_history, max_history = [], [], []
        start = solution_pool.solutions[0].function_value
        min_value, max_value = start, start

        for solution in solution_pool.solutions:
            if solution.function_value < min_value:
                min_value = solution.function_value

            min_history.append(min_value)

            current_history.append(solution.function_value)

            if solution.function_value > max_value:
                max_value = solution.function_value

            max_history.append(max_value)

        self.ax.clear()

        self.ax.plot(min_history, label='min')
        self.ax.plot(max_history, label='max')
        self.ax.plot(current_history, label='current')
        self.ax.legend()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
