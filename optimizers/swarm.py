import numpy as np

from typing import Callable, Tuple, List

from base import BaseOptimizer, Solution, SolutionPool

import time


class Agent:
    def __init__(self, position, personal_velocity, global_velocity, target_function, bounds,
                 minimization: bool = True):
        self.position = np.asarray(position)
        self.personal_velocity = personal_velocity
        self.global_velocity = global_velocity

        self.target_function = target_function

        self.bounds = bounds

        self.known_optimum = None
        self.known_optimum_vector = None

        self.global_known_optimum = None
        self.global_known_optimum_vector = None

        self.minimization = minimization

    def apply_bounds(self, vector):
        for counter, item in enumerate(vector):
            if item < self.bounds[counter][0]:
                vector[counter] = self.bounds[counter][0] + np.random.uniform(0, self.bounds[counter][1]/2)
            elif item > self.bounds[counter][1]:
                vector[counter] = self.bounds[counter][1] - np.random.uniform(0, self.bounds[counter][0]/2)
        return vector

    def do_step(self):

        r1 = np.random.uniform(0, 1)
        r2 = np.random.uniform(0, 1)

        if self.known_optimum is None:
            personal_direction = np.random.uniform(-1, 1, self.position.shape)
        else:
            personal_direction = self.known_optimum_vector - self.position

        if self.global_known_optimum is None:
            global_direction = np.random.uniform(-1, 1, self.position.shape)
        else:
            global_direction = self.global_known_optimum_vector - self.position

        self.position = (self.position + r1 * self.personal_velocity * personal_direction +
                         r2 * self.global_velocity * global_direction)

        self.position = self.position + np.random.uniform(-0.01,0.01, self.position.shape)

        self.position = self.apply_bounds(self.position)

        f = self.target_function(self.position)

        if self.known_optimum_vector is None:
            self.known_optimum_vector = self.position
            self.known_optimum = f
        elif self.minimization:
            if self.known_optimum > f:
                self.known_optimum = f
        else:
            if self.known_optimum < f:
                self.known_optimum = f
        return self.position, f


class SwarmOptimizer(BaseOptimizer):
    def __init__(self, target_function: Callable, bounds: List[Tuple[float, float]],
                 minimization: bool = True, *args, **kwargs):
        super().__init__(target_function, bounds, minimization, *args, **kwargs)

        swarm_params = {'global_velocity': 0.8, 'personal_velocity': 0.21}

        self.swarm_params = swarm_params

        self.swarm_size = 1000
        self.personal_velocity = self.swarm_params["personal_velocity"]
        self.global_velocity = self.swarm_params["global_velocity"]

        self.population = self.create_start_population()

        self.known_optimum = None
        self.known_optimum_vector = None


    def create_start_population(self):
        agents = []
        for _ in range(self.swarm_size):
            vector = []
            for min_val, max_val in self.bounds:
                value = np.random.uniform(min_val, max_val)
                vector.append(value)
            agents.append(
                Agent(vector, self.personal_velocity, self.global_velocity, self.target_function, self.bounds))

        return agents

    def build_bounds(self, bounds):
        return bounds


    def update_global_knowledge(self):
        sorted_agents = sorted(self.population, key=lambda agent: agent.known_optimum)
        if self.minimization:
            optimum = sorted_agents[0].known_optimum
            optimum_vector =sorted_agents[0].position
        else:
            optimum = sorted_agents[0].known_optimum
            optimum_vector = sorted_agents[0].position

        if self.known_optimum_vector is None:
            self.known_optimum_vector = optimum_vector
            self.known_optimum = optimum

        elif self.minimization:
            if self.known_optimum > optimum:
                self.known_optimum = optimum
                self.known_optimum_vector = optimum_vector

        else:
            if self.known_optimum < optimum:
                self.known_optimum = optimum
                self.known_optimum_vector = optimum_vector



    def tell_them_all(self):
        if self.known_optimum is not None:
            for agent in self.population:
                agent.global_known_optimum = self.known_optimum
                agent.global_known_optimum_vector = self.known_optimum_vector

    def optimize(self, rounds, *args, **kwargs):
        for i in range(rounds):
            self.tell_them_all()

            for agent in self.population:
                ask, f = agent.do_step()

                solution = self.create_solution(ask, f)

                self.solution_pool.add_solution(solution)




            self.update_global_knowledge()
