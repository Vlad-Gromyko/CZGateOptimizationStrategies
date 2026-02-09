import json
import pickle
from pathlib import Path
from typing import List
from PySide6.QtCore import QThread, Signal

from core.signals import IOSignals
from core.solution_model import Solution


class IOWorker(QThread):
    def __init__(self):
        super().__init__()
        self.signals = IOSignals()

    def save_solutions(self, solutions: List[Solution], filepath: str, binary: bool = True):
        try:
            if binary:
                with open(filepath, 'wb') as f:
                    pickle.dump(solutions, f)
            else:
                serializable = [
                    {
                        'parameters': sol.parameters,
                        'value': sol.value,
                        'metadata': sol.metadata,
                        'timestamp': sol.timestamp
                    }
                    for sol in solutions
                ]
                with open(filepath, 'w') as f:
                    json.dump(serializable, f, indent=2)

            self.signals.saved.emit()

        except Exception as e:
            self.signals.error.emit(f"Ошибка сохранения: {e}")

    def load_solutions(self, filepath: str, binary: bool = True):
        try:
            if binary:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                data = [
                    Solution(
                        parameters=sol['parameters'],
                        value=sol['value'],
                        metadata=sol.get('metadata'),
                        timestamp=sol.get('timestamp')
                    )
                    for sol in data
                ]

            self.signals.loaded.emit(data)

        except Exception as e:
            self.signals.error.emit(f"Ошибка загрузки: {e}")