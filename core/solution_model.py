from dataclasses import dataclass
from typing import List, Optional
import time


@dataclass
class Solution:
    parameters: List[float]
    value: float
    metadata: Optional[dict] = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class SolutionModel:
    def __init__(self):
        self.solutions: List[Solution] = []

    def add_solution(self, solution: Solution):
        self.solutions.append(solution)

    def clear(self):
        self.solutions.clear()

    def get_all(self):
        return self.solutions.copy()

    def get_best(self, n=1):
        sorted_solutions = sorted(self.solutions, key=lambda x: x.value)
        return sorted_solutions[:n]