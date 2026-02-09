from PySide6.QtCore import QThread, QMutex, QWaitCondition
from typing import Callable

from core.signals import ComputeSignals
from core.solution_model import Solution
from core.compute_context import ComputeContext


class ComputeWorker(QThread):
    def __init__(self):
        super().__init__()
        self.signals = ComputeSignals()
        self._mutex = QMutex()
        self._condition = QWaitCondition()

        # Контекст для связи с функцией
        self.context = ComputeContext(self.signals)

        # Контроль выполнения
        self._is_running = False
        self._is_paused = False
        self._should_stop = False

        # Текущая задача
        self._compute_func: Callable = None
        self._func_params: list = None

    def set_task(self, func: Callable, params: list):
        """Установить задачу для выполнения"""
        self._compute_func = func
        self._func_params = params

    def run(self):
        """Основной метод выполнения задачи"""
        self._should_stop = False
        self.context.set_stop_flag(False)
        self.context.set_pause_flag(False)
        self._is_running = True

        self.signals.started.emit()

        if not self._compute_func:
            self.signals.error.emit("Функция не установлена")
            return

        try:
            # Вызываем функцию с контекстом
            self._compute_func(self._func_params, self.context)

            # Сигнал о завершении
            self.signals.finished.emit()

        except Exception as e:
            self.signals.error.emit(f"Ошибка вычислений: {str(e)}")
        finally:
            self._is_running = False

    def pause(self):
        """Поставить вычисление на паузу"""
        if self._is_running and not self._is_paused:
            self._is_paused = True
            self.context.set_pause_flag(True)
            self.signals.paused.emit()

    def resume(self):
        """Возобновить вычисление"""
        if self._is_paused:
            self._is_paused = False
            self.context.set_pause_flag(False)
            self.signals.resumed.emit()

    def stop(self):
        """Остановить вычисление"""
        self._should_stop = True
        self.context.set_stop_flag(True)
        # Если на паузе, снимаем паузу для корректной остановки
        if self._is_paused:
            self.resume()
        self.signals.stopped.emit()

    def is_running(self):
        """Проверка, выполняется ли вычисление"""
        return self._is_running

    def is_paused(self):
        """Проверка, на паузе ли вычисление"""
        return self._is_paused