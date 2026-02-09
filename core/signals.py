from PySide6.QtCore import Signal, QObject

class ComputeSignals(QObject):
    progress = Signal(int, str)  # процент, статус
    solution_ready = Signal(object)  # готовый Solution объект
    error = Signal(str)
    paused = Signal()
    resumed = Signal()
    stopped = Signal()
    started = Signal()
    finished = Signal()  # вычисления завершены

class IOSignals(QObject):
    loaded = Signal(list)  # список решений
    saved = Signal()
    error = Signal(str)