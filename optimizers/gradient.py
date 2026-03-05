import numpy as np
from typing import Callable, Tuple, List
import math
from base import BaseOptimizer, Solution, SolutionPool

import os

import time


class AdamWL2Optimizer(BaseOptimizer):
    def __init__(self, target_function: Callable, bounds: List[Tuple[float, float]],
                 minimization: bool = True, *args, **kwargs):
        super().__init__(target_function, bounds, minimization, *args, **kwargs)

        self.x = np.random.uniform(5, 6, len(self.bounds))

        self.x = self.apply_bounds(self.x)

        self.x_history = []

        self.gamma = 0.05
        self.step = 0.0005

        self.beta_1 = 0.99
        self.beta_2 = 0.999

        self.gradient = 0

        self.m = 0
        self.v = 0

        self.m_hat = 0
        self.v_hat = 0

        self.g = 0

        self.l2 = 0
        self.epsilon = 0.00000001
        self._lambda = 0.15

        self.gradient_centralization = False

        self.gradient_type = 'stochastic'
        self.steps_distribution = 'Uniform'

    def find_gradient(self):
        if self.gradient_type == 'stochastic':
            return self.stochastic_gradient()
        return 0

    def full_gradient(self):

        gradient = np.zeros_like(self.x)

        for i in range(len(self.x)):
            u_minus = self.x.copy()
            u_plus = self.x.copy()

            u_minus[i] -= self.step
            u_plus[i] += self.step

            m_plus = self.target_function(u_plus)

            m_minus = self.target_function(u_minus)

            gradient[i] = (m_plus - m_minus)

        return gradient

    def calc_steps(self):
        num = len(self.x)
        steps = np.zeros(num)

        if self.steps_distribution == 'Bernoulli':
            steps = np.random.choice([-1, 1], size=num)

        elif self.steps_distribution == 'Uniform':
            steps = np.random.uniform(-1, 1, size=num)

        elif self.steps_distribution == 'Coordinate':
            steps[np.random.randint(num)] = np.random.choice([-1, 1])

        return steps

    def stochastic_gradient(self):
        steps = self.calc_steps()

        num_zeros = np.random.randint(0, len(steps)//2)
        indices = np.random.choice(len(steps), size=num_zeros, replace=False)

        steps[indices] = 0

        u_plus = self.apply_bounds(self.x + steps * self.step)
        u_minus = self.apply_bounds(self.x - steps * self.step)
        m_plus = self.target_function(u_plus)

        solution = self.create_solution(u_plus, m_plus)

        self.solution_pool.add_solution(solution)

        m_minus = self.target_function(u_minus)

        solution = self.create_solution(u_minus, m_minus)

        self.solution_pool.add_solution(solution)

        gradient = (m_plus - m_minus) * steps / self.step / 2

        return gradient

    def optimize(self, rounds, *args, **kwargs):
        if self.minimization:
            multiply = 1
        else:
            multiply = -1
        for i in range(rounds):
            self.gradient = self.find_gradient()
            print(self.gradient, 'GRADIENT')

            if self.gradient_centralization:
                self.gradient = self.gradient - np.mean(self.gradient)


            if len(self.x_history) != 0:
                self.g = self.gradient +  self.x_history[-1] * self.l2
                self.m = self.beta_1 * self.m + (1 - self.beta_1) * self.g
                self.v = self.beta_2 * self.v + (1 - self.beta_2) * self.g ** 2
            else:
                self.g = self.gradient
                self.m = self.gradient
                self.v = self.gradient**2

            self.m_hat = self.m / (1 - self.beta_1 ** (i + 1))
            self.v_hat = self.v / (1 - self.beta_2 ** (i + 1))

            self.x = self.x - multiply * self.gamma * self.m_hat / (np.sqrt(self.v_hat) + self.epsilon)

            if len(self.x_history) != 0:
                self.x = self.x - multiply * self.gamma * self._lambda * self.x_history[-1]

            self.x = self.apply_bounds(self.x)

            self.x_history.append(self.x)

    def build_bounds(self, bounds):
        return bounds

    def apply_bounds(self, vector):

        for counter, item in enumerate(vector):
            if item < self.bounds[counter][0]:
                vector[counter] = self.bounds[counter][0] + np.random.uniform(0, self.bounds[counter][1]/10000) * self.step
            elif item > self.bounds[counter][1]:
                vector[counter] = self.bounds[counter][1]- np.random.uniform(0, self.bounds[counter][1]/10000) * self.step
        return vector
