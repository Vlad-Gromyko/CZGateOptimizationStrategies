from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QFont, QFontMetrics
import numpy as np
from typing import List


class ParallelCoordinatesWidget(QWidget):
    """Виджет для отображения решений в параллельных координатах"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.solutions = []
        self.axes = []
        self.normalized_data = []

        self._init_parameters()
        self._init_ui()

    def _init_ui(self):
        self.setMinimumSize(600, 400)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Панель управления
        control_layout = QHBoxLayout()

        control_layout.addWidget(QLabel("Оси:"))
        self.axis_select = QComboBox()
        self.axis_select.addItems(["Автоматически", "Ручная настройка"])
        control_layout.addWidget(self.axis_select)

        self.btn_refresh = QPushButton("Обновить")
        self.btn_refresh.clicked.connect(self.update_display)
        control_layout.addWidget(self.btn_refresh)

        control_layout.addStretch()

        layout.addLayout(control_layout)
        layout.addStretch()

    def _init_parameters(self):
        self.margin = 70
        self.top_margin = 40
        self.bottom_margin = 40
        self.axis_width = 2
        self.line_width = 1
        self.point_size = 5

        # Цвета
        self.axis_color = QColor(80, 80, 80)
        self.text_color = QColor(30, 30, 30)
        self.grid_color = QColor(220, 220, 220)

        # Цвета для решений (по значению)
        self.solution_colors = [
            QColor(255, 100, 100, 150),  # Красный для плохих значений
            QColor(255, 200, 100, 150),  # Оранжевый
            QColor(255, 255, 100, 150),  # Желтый
            QColor(100, 255, 100, 150),  # Зеленый для хороших значений
        ]

        self.bg_color = QColor(250, 250, 250)

    def set_solutions(self, solutions: List):
        """Установить решения для отображения"""
        self.solutions = solutions
        if solutions:
            self.update_axes()
            self.update_display()

    def update_axes(self):
        """Обновление осей на основе данных"""
        if not self.solutions:
            return

        # Определяем количество параметров в первом решении
        first_solution = self.solutions[0]
        num_params = len(first_solution.parameters)

        self.axes = []
        for i in range(num_params):
            axis_name = f"X{i + 1}"
            # Находим min/max значения для этого параметра
            values = [sol.parameters[i] for sol in self.solutions]
            axis = {
                'name': axis_name,
                'min': min(values),
                'max': max(values),
                'index': i
            }
            self.axes.append(axis)

        self.normalize_data()

    def normalize_data(self):
        """Нормализация данных для отображения"""
        if not self.solutions or not self.axes:
            return

        self.normalized_data = []

        for solution in self.solutions:
            normalized = []
            for i, axis in enumerate(self.axes):
                value = solution.parameters[i]
                # Нормализация к [0, 1]
                if axis['max'] != axis['min']:
                    norm_value = (value - axis['min']) / (axis['max'] - axis['min'])
                else:
                    norm_value = 0.5
                normalized.append(norm_value)
            self.normalized_data.append(normalized)

    def update_display(self):
        """Обновление отображения"""
        self.update()

    def get_solution_color(self, value, min_val, max_val):
        """Получить цвет решения на основе значения"""
        if max_val == min_val:
            return self.solution_colors[0]

        # Нормализуем значение от 0 до 1
        norm = (value - min_val) / (max_val - min_val)

        # Выбираем цвет из градиента
        color_index = int(norm * (len(self.solution_colors) - 1))
        color_index = max(0, min(color_index, len(self.solution_colors) - 1))

        return self.solution_colors[color_index]

    def paintEvent(self, event):
        """Отрисовка виджета"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()

        # Очистка фона
        painter.fillRect(0, 0, width, height, QBrush(self.bg_color))

        if not self.solutions or not self.normalized_data:
            # Рисуем сообщение об отсутствии данных
            painter.setPen(self.text_color)
            painter.setFont(QFont("Arial", 12))
            painter.drawText(
                width // 2 - 100,
                height // 2,
                "Нет данных для отображения"
            )
            painter.end()
            return

        # Рисование осей
        num_axes = len(self.axes)
        if num_axes < 2:
            painter.setPen(self.text_color)
            painter.drawText(
                width // 2 - 100,
                height // 2,
                "Недостаточно параметров для отображения"
            )
            painter.end()
            return

        # Расстояние между осями
        axis_spacing = (width - 2 * self.margin) / (num_axes - 1)

        # Рисуем оси и подписи
        axis_positions = []
        for i, axis in enumerate(self.axes):
            x = self.margin + i * axis_spacing

            # Ось
            painter.setPen(QPen(self.axis_color, self.axis_width))
            painter.drawLine(
                x, self.top_margin,
                x, height - self.bottom_margin
            )

            # Подпись оси
            painter.setPen(self.text_color)
            painter.setFont(QFont("Arial", 9, QFont.Bold))
            text_width = QFontMetrics(painter.font()).horizontalAdvance(axis['name'])
            painter.drawText(
                x - text_width // 2,
                self.top_margin - 10,
                axis['name']
            )

            # Деления и значения
            painter.setFont(QFont("Arial", 8))

            # Верхнее значение (максимум)
            max_text = f"{axis['max']:.2f}"
            text_width = QFontMetrics(painter.font()).horizontalAdvance(max_text)
            painter.setPen(QPen(self.axis_color, 1))
            painter.drawLine(x - 5, self.top_margin, x + 5, self.top_margin)
            painter.setPen(self.text_color)
            painter.drawText(x - text_width // 2, self.top_margin - 20, max_text)

            # Нижнее значение (минимум)
            min_text = f"{axis['min']:.2f}"
            text_width = QFontMetrics(painter.font()).horizontalAdvance(min_text)
            painter.setPen(QPen(self.axis_color, 1))
            painter.drawLine(x - 5, height - self.bottom_margin, x + 5, height - self.bottom_margin)
            painter.setPen(self.text_color)
            painter.drawText(
                x - text_width // 2,
                height - self.bottom_margin + 15,
                min_text
            )

            # Сетка (необязательно, но полезно)
            painter.setPen(QPen(self.grid_color, 0.5))
            for j in range(1, 4):  # 3 линии сетки
                y = self.top_margin + (height - self.top_margin - self.bottom_margin) * j / 4
                painter.drawLine(
                    self.margin - 10, y,
                    width - self.margin + 10, y
                )

            axis_positions.append(x)

        # Рисуем линии решений
        if self.solutions:
            # Находим min и max значений функции для градиента цвета
            values = [sol.value for sol in self.solutions]
            min_val = min(values) if values else 0
            max_val = max(values) if values else 1

            for idx, (solution, solution_data) in enumerate(zip(self.solutions, self.normalized_data)):
                # Цвет в зависимости от значения функции
                color = self.get_solution_color(solution.value, min_val, max_val)

                painter.setPen(QPen(color, self.line_width))

                points = []
                for i, norm_value in enumerate(solution_data):
                    x = axis_positions[i]
                    # Преобразуем нормализованное значение в координату Y
                    y = height - self.bottom_margin - norm_value * (height - self.top_margin - self.bottom_margin)
                    points.append((x, y))

                # Рисуем полилинию
                for j in range(len(points) - 1):
                    painter.drawLine(
                        int(points[j][0]), int(points[j][1]),
                        int(points[j + 1][0]), int(points[j + 1][1])
                    )

                # Рисуем точки на осях (только для первых 20 решений, чтобы не перегружать)
                if idx < 20:
                    painter.setBrush(QBrush(color))
                    for x, y in points:
                        painter.drawEllipse(
                            int(x) - self.point_size // 2,
                            int(y) - self.point_size // 2,
                            self.point_size, self.point_size
                        )

        # Легенда (цветовая шкала)
        legend_height = 20
        legend_top = height - 10
        legend_width = 200
        legend_left = width - legend_width - 10

        painter.setPen(QPen(self.text_color, 1))
        painter.drawRect(legend_left, legend_top - legend_height, legend_width, legend_height)

        # Градиент в легенде
        for i in range(legend_width):
            color_index = int(i / legend_width * (len(self.solution_colors) - 1))
            color_index = max(0, min(color_index, len(self.solution_colors) - 1))
            color = self.solution_colors[color_index]
            painter.setPen(QPen(color, 1))
            painter.drawLine(legend_left + i, legend_top - legend_height,
                             legend_left + i, legend_top)

        # Подписи к легенде
        painter.setPen(self.text_color)
        painter.setFont(QFont("Arial", 8))
        painter.drawText(legend_left, legend_top + 15, f"{min_val:.2f}")

        max_text = f"{max_val:.2f}"
        text_width = QFontMetrics(painter.font()).horizontalAdvance(max_text)
        painter.drawText(legend_left + legend_width - text_width, legend_top + 15, max_text)

        painter.setFont(QFont("Arial", 9, QFont.Bold))
        legend_title = "Значение функции"
        text_width = QFontMetrics(painter.font()).horizontalAdvance(legend_title)
        painter.drawText(legend_left + (legend_width - text_width) // 2,
                         legend_top - legend_height - 5, legend_title)

        painter.end()

    def resizeEvent(self, event):
        """Обработчик изменения размера виджета"""
        self.update_display()
        super().resizeEvent(event)