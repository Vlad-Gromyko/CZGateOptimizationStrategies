import time
import random
from typing import Callable
from core.signals import ComputeSignals


class ComputeContext:
    """
    Контекст вычислений, передаваемый в функцию.
    Через него функция может отправлять сигналы в интерфейс.
    """

    def __init__(self, signals: ComputeSignals):
        self.signals = signals
        self._should_stop = False
        self._is_paused = False
        self._pause_flag = False  # Флаг для блокировки при паузе

    def send_progress(self, percent: int, message: str = ""):
        """Отправить прогресс вычислений"""
        self.signals.progress.emit(percent, message)

    def send_solution(self, solution):
        """Отправить готовое решение"""
        self.signals.solution_ready.emit(solution)

    def send_error(self, error_message: str):
        """Отправить ошибку"""
        self.signals.error.emit(error_message)

    def check_stop(self) -> bool:
        """Проверить, запрошена ли остановка"""
        return self._should_stop

    def check_pause(self):
        """Проверить паузу - метод для использования в функциях"""
        while self._is_paused and not self._should_stop:
            time.sleep(0.1)  # Короткая пауза
            # Отправляем сигнал о том, что мы на паузе
            self.signals.progress.emit(-1, "Пауза...")

    def set_stop_flag(self, value: bool):
        """Установить флаг остановки"""
        self._should_stop = value

    def set_pause_flag(self, value: bool):
        """Установить флаг паузы"""
        self._is_paused = value


# Примеры функций, которые проверяют паузу и остановку
def example_function_1(params: list, context: ComputeContext):
    """Пример функции, которая создает несколько решений и реагирует на паузу"""
    import time

    steps = 20
    for i in range(steps):
        # Проверка остановки
        if context.check_stop():
            context.send_progress(0, "Вычисления остановлены")
            return

        # Проверка паузы (блокирует выполнение, если пауза установлена)
        context.check_pause()

        # Имитация вычислений
        time.sleep(0.1)

        # Создание решения
        value = sum(x ** 2 for x in params) * (i + 1) / steps
        solution = {
            'parameters': params.copy(),
            'value': value,
            'metadata': {'iteration': i, 'total_steps': steps}
        }

        # Отправка решения через контекст
        context.send_solution(solution)

        # Отправка прогресса
        progress = int((i + 1) * 100 / steps)
        context.send_progress(progress, f"Создано решений: {i + 1}")

    context.send_progress(100, "Вычисления завершены")


def example_function_2(params: list, context: ComputeContext):
    """Пример функции, которая создает несколько решений и реагирует на паузу"""
    import time


    if context.check_stop():
        context.send_progress(0, "Вычисления остановлены")
        return

        # Проверка паузы (блокирует выполнение, если пауза установлена)
    context.check_pause()

        # Имитация вычислений
    time.sleep(0.1)

        # Создание решения
    value = sum(x for x in params)
    solution = {
            'parameters': params.copy(),
            'value': value,
            'metadata': {'iteration': 0, 'total_steps': 0}
        }

        # Отправка решения через контекст
    context.send_solution(solution)

        # Отправка прогресса

    context.send_progress(100, "Вычисления завершены")


def monte_carlo_simulation(params: list, context: ComputeContext):
    """Монте-Карло симуляция с постепенным улучшением результата"""
    import time

    num_iterations = 100
    best_value = float('inf')
    best_params = params.copy()

    for i in range(num_iterations):
        # Проверка остановки
        if context.check_stop():
            context.send_progress(0, "Вычисления остановлены")
            return

        # Проверка паузы
        context.check_pause()

        time.sleep(0.05)

        # Генерация нового набора параметров
        new_params = [
            p + random.uniform(-0.5, 0.5) for p in params
        ]

        # Вычисление значения функции
        value = sum(x ** 2 for x in new_params)

        # Обновление лучшего результата
        if value < best_value:
            best_value = value
            best_params = new_params.copy()

            # Отправка улучшенного решения
            solution = {
                'parameters': best_params.copy(),
                'value': best_value,
                'metadata': {
                    'iteration': i,
                    'improvement': True,
                    'algorithm': 'monte_carlo'
                }
            }
            context.send_solution(solution)

        # Отправка прогресса
        progress = int((i + 1) * 100 / num_iterations)
        if i % 10 == 0:
            context.send_progress(progress, f"Итерация {i + 1}, лучшее: {best_value:.6f}")
        else:
            context.send_progress(progress, f"Итерация {i + 1}")
