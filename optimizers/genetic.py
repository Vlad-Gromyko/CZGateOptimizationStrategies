import pygad

from typing import Callable, Tuple, List

from base import BaseOptimizer, Solution, SolutionPool


class GeneticOptimizer(BaseOptimizer):
    def __init__(self, target_function: Callable, bounds: List[Tuple[float, float]],
                 minimization: bool = True, *args, **kwargs):
        super().__init__(target_function, bounds, minimization, *args, **kwargs)

        if self.minimization:
            fitness_function = lambda ga, vector, idx :target_function(vector) * -1
        else:
            fitness_function = lambda ga, vector, idx :target_function(vector)

        num_generations = 500
        num_parents_mating = 50

        sol_per_pop = 500
        num_genes = len(bounds)

        gen_space = self.bounds

        parent_selection_type = "sss"
        keep_parents = 5

        crossover_type = "single_point"

        mutation_type = "random"
        mutation_percent_genes = 50


        def on_start(ga_instance):
            population = ga_instance.population
            fitness_values = ga_instance.cal_pop_fitness()
            for ind, fit in zip(population, fitness_values):
                if self.minimization:
                    solution = self.create_solution(ind, -1 * fit)
                else:
                    solution = self.create_solution(ind, fit)
                self.solution_pool.add_solution(solution)

        def on_generation(ga_instance):
            population = ga_instance.population
            fitnesses = ga_instance.last_generation_fitness


            for ind, fit in zip(population, fitnesses):
                if self.minimization:
                    solution = self.create_solution(ind, -1 * fit)
                else:
                    solution = self.create_solution(ind, fit)
                self.solution_pool.add_solution(solution)

        self.ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       gene_space=gen_space,
                       on_start=on_start,
                       on_generation=on_generation,)

    def optimize(self, rounds, *args, **kwargs):
        self.ga_instance.num_generations = rounds

        self.ga_instance.run()

    def build_bounds(self, bounds: List[Tuple[float, float]]):
        return [{'low': bound[0], 'high': bound[1]} for bound in bounds]


