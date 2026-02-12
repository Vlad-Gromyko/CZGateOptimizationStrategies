from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QListWidgetItem,
    QGroupBox, QLineEdit, QLabel, QProgressBar,
    QTabWidget, QTextEdit, QFileDialog, QMessageBox,
    QComboBox, QSpinBox, QDoubleSpinBox, QFormLayout,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QCheckBox, QScrollArea, QSplitter, QTreeWidget,
    QTreeWidgetItem, QDockWidget, QStatusBar,
    QApplication
)
import numpy as np
from PySide6.QtCore import Qt, Slot, QTimer, QSize
from PySide6.QtGui import QFont
import json
import time

from core.signals import ComputeSignals, IOSignals
from core.solution_model import Solution, SolutionModel
from core.workers.compute_worker import ComputeWorker
from core.workers.io_worker import IOWorker
from core.compute_context import example_function_1, example_function_2, monte_carlo_simulation
from utils.parallel_coordinates import ParallelCoordinatesWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Система оптимизации")
        self.setGeometry(100, 100, 1400, 800)

        # Модель данных
        self.solution_model = SolutionModel()

        # Потоки
        self.compute_worker = ComputeWorker()
        self.io_worker = IOWorker()

        # Список доступных функций
        self.available_functions = {
            "Пример функции 1 (пошаговая)": example_function_1,
            "Пример функции 2 (пошаговая)": example_function_2,
            "Монте-Карло симуляция": monte_carlo_simulation,
        }

        # Таймер для обновления статистики
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats)

        # Инициализация UI
        self._init_ui()

        # Подключение сигналов (после инициализации UI)
        self._connect_signals()

        # Статус бар
        self.statusBar().showMessage("Готов")

    def _init_ui(self):
        # Центральный виджет с вкладками
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # ========== ВЕРХНЯЯ ПАНЕЛЬ: ПРОГРЕСС И УПРАВЛЕНИЕ ==========
        control_panel = QGroupBox("Управление вычислениями")
        control_layout = QVBoxLayout()

        # Кнопки управления
        button_layout = QHBoxLayout()

        self.btn_start = QPushButton("▶ Запуск")
        self.btn_pause = QPushButton("⏸ Пауза")
        self.btn_stop = QPushButton("⏹ Стоп")
        self.btn_clear = QPushButton("🗑 Очистить")
        self.btn_save = QPushButton("💾 Сохранить")
        self.btn_load = QPushButton("📂 Загрузить")

        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)

        # Стилизация кнопок
        for btn in [self.btn_start, self.btn_pause, self.btn_stop,
                    self.btn_clear, self.btn_save, self.btn_load]:
            btn.setMinimumHeight(30)

        button_layout.addWidget(self.btn_start)
        button_layout.addWidget(self.btn_pause)
        button_layout.addWidget(self.btn_stop)
        button_layout.addWidget(self.btn_clear)
        button_layout.addWidget(self.btn_save)
        button_layout.addWidget(self.btn_load)
        button_layout.addStretch()

        control_layout.addLayout(button_layout)

        # Прогресс-бар
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Прогресс:"))

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(20)
        self.progress_bar.setTextVisible(True)

        self.status_label = QLabel("Готов к работе")
        self.status_label.setMinimumWidth(200)

        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)

        control_layout.addLayout(progress_layout)

        # Статистика
        stats_layout = QHBoxLayout()
        self.stats_label = QLabel("Решений: 0 | Лучшее: N/A")
        stats_layout.addWidget(self.stats_label)
        stats_layout.addStretch()

        control_layout.addLayout(stats_layout)

        control_panel.setLayout(control_layout)
        main_layout.addWidget(control_panel)

        # ========== ЦЕНТРАЛЬНЫЙ TAB WIDGET ==========
        self.tab_widget = QTabWidget()

        # Вкладка 1: Параметры целевой функции
        self.function_params_tab = self._create_function_params_tab()
        self.tab_widget.addTab(self.function_params_tab, "🎯 Целевая функция")

        # Вкладка 2: Пространство параметров
        self.parameter_space_tab = self._create_parameter_space_tab()
        self.tab_widget.addTab(self.parameter_space_tab, "📊 Пространство параметров")

        # Вкладка 3: Каталог решений
        self.solutions_tab = self._create_solutions_tab()
        self.tab_widget.addTab(self.solutions_tab, "📁 Каталог решений")

        # Вкладка 4: Параллельные координаты
        self.parallel_coords_tab = self._create_parallel_coords_tab()
        self.tab_widget.addTab(self.parallel_coords_tab, "📈 Параллельные координаты")

        # Вкладка 5: Лог
        self.log_tab = self._create_log_tab()
        self.tab_widget.addTab(self.log_tab, "📝 Лог")

        main_layout.addWidget(self.tab_widget)

    def _create_function_params_tab(self):
        """Создание вкладки параметров целевой функции"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Группа: Выбор функции
        function_group = QGroupBox("Выбор целевой функции")
        function_layout = QVBoxLayout()

        func_layout = QHBoxLayout()
        func_layout.addWidget(QLabel("Функция:"))
        self.func_combo = QComboBox()
        self.func_combo.addItems(self.available_functions.keys())
        self.func_combo.setMinimumWidth(300)
        func_layout.addWidget(self.func_combo)
        function_layout.addLayout(func_layout)

        # Описание функции
        self.func_description = QTextEdit()
        self.func_description.setReadOnly(True)
        self.func_description.setMaximumHeight(100)
        self.func_description.setText("Описание выбранной функции будет здесь...")
        function_layout.addWidget(QLabel("Описание:"))
        function_layout.addWidget(self.func_description)

        function_group.setLayout(function_layout)
        scroll_layout.addWidget(function_group)

        # Группа: Параметры функции
        params_group = QGroupBox("Параметры функции")
        params_layout = QVBoxLayout()

        # Количество параметров
        count_layout = QHBoxLayout()
        count_layout.addWidget(QLabel("Количество параметров:"))
        self.param_count_spin = QSpinBox()
        self.param_count_spin.setRange(1, 20)
        self.param_count_spin.setValue(3)
        self.param_count_spin.valueChanged.connect(self._update_param_controls)
        count_layout.addWidget(self.param_count_spin)
        count_layout.addStretch()
        params_layout.addLayout(count_layout)

        # Динамические поля параметров
        self.param_widget = QWidget()
        self.param_layout = QFormLayout(self.param_widget)
        params_layout.addWidget(self.param_widget)

        params_group.setLayout(params_layout)
        scroll_layout.addWidget(params_group)

        # Группа: Настройки вычислений
        settings_group = QGroupBox("Настройки вычислений")
        settings_layout = QFormLayout()

        self.max_iterations = QSpinBox()
        self.max_iterations.setRange(10, 10000)
        self.max_iterations.setValue(100)
        settings_layout.addRow("Максимальное число итераций:", self.max_iterations)

        self.timeout = QDoubleSpinBox()
        self.timeout.setRange(0, 3600)
        self.timeout.setValue(60)
        self.timeout.setSuffix(" сек")
        settings_layout.addRow("Таймаут вычислений:", self.timeout)

        settings_group.setLayout(settings_layout)
        scroll_layout.addWidget(settings_group)

        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)

        # Инициализация параметров (но без вызова _update_param_bounds_table)
        self._init_param_controls()

        return widget

    def _init_param_controls(self):
        """Инициализация элементов управления параметрами (без вызова _update_param_bounds_table)"""
        # Очистка старых элементов
        while self.param_layout.rowCount() > 0:
            self.param_layout.removeRow(0)

        # Создание новых элементов
        param_count = self.param_count_spin.value()
        self.param_spinboxes = []

        for i in range(param_count):
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-1000, 1000)
            spinbox.setValue(1.0 if i % 2 == 0 else -1.0)
            spinbox.setDecimals(6)
            self.param_layout.addRow(f"Параметр {i + 1}:", spinbox)
            self.param_spinboxes.append(spinbox)

    def _create_parameter_space_tab(self):
        """Создание вкладки пространства параметров"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Таблица границ параметров
        self.param_bounds_table = QTableWidget()
        self.param_bounds_table.setColumnCount(5)
        self.param_bounds_table.setHorizontalHeaderLabels([
            "Параметр", "Минимум", "Максимум", "Шаг", "Тип"
        ])
        self.param_bounds_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        layout.addWidget(self.param_bounds_table)

        # Кнопки управления
        bounds_buttons = QHBoxLayout()
        self.btn_update_bounds = QPushButton("Обновить границы")
        self.btn_auto_bounds = QPushButton("Автоматические границы")
        self.btn_reset_bounds = QPushButton("Сбросить")

        bounds_buttons.addWidget(self.btn_update_bounds)
        bounds_buttons.addWidget(self.btn_auto_bounds)
        bounds_buttons.addWidget(self.btn_reset_bounds)
        bounds_buttons.addStretch()

        layout.addLayout(bounds_buttons)

        # Заполняем таблицу текущими значениями
        self._update_param_bounds_table()

        return widget

    def _update_param_controls(self):
        """Обновление элементов управления параметрами"""
        # Очистка старых элементов
        while self.param_layout.rowCount() > 0:
            self.param_layout.removeRow(0)

        # Создание новых элементов
        param_count = self.param_count_spin.value()
        self.param_spinboxes = []

        for i in range(param_count):
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-1000, 1000)
            spinbox.setValue(1.0 if i % 2 == 0 else -1.0)
            spinbox.setDecimals(6)
            self.param_layout.addRow(f"Параметр {i + 1}:", spinbox)
            self.param_spinboxes.append(spinbox)

        # Обновление таблицы границ параметров, только если она уже создана
        if hasattr(self, 'param_bounds_table'):
            self._update_param_bounds_table()

    def _update_param_bounds_table(self):
        """Обновление таблицы границ параметров"""
        if not hasattr(self, 'param_bounds_table'):
            return

        param_count = self.param_count_spin.value()
        self.param_bounds_table.setRowCount(param_count)

        for i in range(param_count):
            # Параметр
            self.param_bounds_table.setItem(i, 0, QTableWidgetItem(f"Параметр {i + 1}"))

            # Минимум
            min_item = QTableWidgetItem("-10.0")
            min_item.setFlags(min_item.flags() | Qt.ItemIsEditable)
            self.param_bounds_table.setItem(i, 1, min_item)

            # Максимум
            max_item = QTableWidgetItem("10.0")
            max_item.setFlags(max_item.flags() | Qt.ItemIsEditable)
            self.param_bounds_table.setItem(i, 2, max_item)

            # Шаг
            step_item = QTableWidgetItem("0.1")
            step_item.setFlags(step_item.flags() | Qt.ItemIsEditable)
            self.param_bounds_table.setItem(i, 3, step_item)

            # Тип
            type_combo = QComboBox()
            type_combo.addItems(["Вещественный", "Целочисленный", "Логический"])
            self.param_bounds_table.setCellWidget(i, 4, type_combo)

    def _create_solutions_tab(self):
        """Создание вкладки каталога решений"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        splitter = QSplitter(Qt.Horizontal)

        # Левая панель: список решений
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Фильтры и поиск
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Фильтр:"))
        self.solution_filter = QComboBox()
        self.solution_filter.addItems(["Все", "Только лучшие", "Последние 50"])
        filter_layout.addWidget(self.solution_filter)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Поиск по параметрам...")
        filter_layout.addWidget(self.search_box)

        left_layout.addLayout(filter_layout)

        # Таблица решений
        self.solutions_table = QTableWidget()
        self.solutions_table.setColumnCount(6)
        self.solutions_table.setHorizontalHeaderLabels([
            "ID", "Значение", "Параметры", "Время", "Статус", "Действия"
        ])
        self.solutions_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.solutions_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.solutions_table.setAlternatingRowColors(True)

        left_layout.addWidget(self.solutions_table)

        # Правая панель: детали решения
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        details_group = QGroupBox("Детали решения")
        details_layout = QFormLayout()

        self.detail_id = QLabel("-")
        self.detail_value = QLabel("-")
        self.detail_params = QLabel("-")
        self.detail_time = QLabel("-")
        self.detail_metadata = QTextEdit()
        self.detail_metadata.setReadOnly(True)
        self.detail_metadata.setMaximumHeight(150)

        details_layout.addRow("ID:", self.detail_id)
        details_layout.addRow("Значение:", self.detail_value)
        details_layout.addRow("Параметры:", self.detail_params)
        details_layout.addRow("Время:", self.detail_time)
        details_layout.addRow("Метаданные:", self.detail_metadata)

        details_group.setLayout(details_layout)
        right_layout.addWidget(details_group)

        # Кнопки действий
        action_buttons = QHBoxLayout()
        self.btn_export_solution = QPushButton("Экспорт")
        self.btn_compare = QPushButton("Сравнить")
        self.btn_visualize = QPushButton("Визуализировать")

        action_buttons.addWidget(self.btn_export_solution)
        action_buttons.addWidget(self.btn_compare)
        action_buttons.addWidget(self.btn_visualize)

        right_layout.addLayout(action_buttons)
        right_layout.addStretch()

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([700, 300])

        layout.addWidget(splitter)

        return widget

    def _create_parallel_coords_tab(self):
        """Создание вкладки параллельных координат"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Виджет параллельных координат
        self.parallel_coords = ParallelCoordinatesWidget()
        layout.addWidget(self.parallel_coords)

        # Настройки визуализации
        settings_group = QGroupBox("Настройки визуализации")
        settings_layout = QHBoxLayout()

        settings_layout.addWidget(QLabel("Количество решений:"))
        self.viz_count = QSpinBox()
        self.viz_count.setRange(10, 1000)
        self.viz_count.setValue(50)
        settings_layout.addWidget(self.viz_count)

        settings_layout.addWidget(QLabel("Цветовая схема:"))
        self.color_scheme = QComboBox()
        self.color_scheme.addItems(["По значению", "По времени", "По категории"])
        settings_layout.addWidget(self.color_scheme)

        settings_layout.addWidget(QLabel("Толщина линий:"))
        self.line_width = QSpinBox()
        self.line_width.setRange(1, 5)
        self.line_width.setValue(2)
        settings_layout.addWidget(self.line_width)

        self.btn_update_viz = QPushButton("Обновить график")
        self.btn_update_viz.clicked.connect(self.update_visualization)
        settings_layout.addWidget(self.btn_update_viz)

        settings_layout.addStretch()
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        return widget

    @Slot()
    def update_visualization(self):
        """Обновление визуализации"""
        if hasattr(self, 'parallel_coords') and self.solution_model.solutions:
            # Берем последние N решений
            count = self.viz_count.value()
            solutions_to_show = self.solution_model.solutions[-count:] if count < len(
                self.solution_model.solutions) else self.solution_model.solutions
            self.parallel_coords.set_solutions(solutions_to_show)
            self.log_message(f"Визуализация обновлена: показано {len(solutions_to_show)} решений")
        else:
            self.log_message("Нет данных для визуализации")

    def _create_log_tab(self):
        """Создание вкладки лога"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Панель управления логом
        log_control = QHBoxLayout()

        self.btn_clear_log = QPushButton("Очистить лог")
        self.btn_save_log = QPushButton("Сохранить лог")
        self.log_level = QComboBox()
        self.log_level.addItems(["Все", "Только ошибки", "Только предупреждения"])

        log_control.addWidget(self.btn_clear_log)
        log_control.addWidget(self.btn_save_log)
        log_control.addWidget(QLabel("Уровень:"))
        log_control.addWidget(self.log_level)
        log_control.addStretch()

        layout.addLayout(log_control)

        # Текстовое поле лога
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))

        layout.addWidget(self.log_text)

        return widget

    def _connect_signals(self):
        """Подключение всех сигналов и слотов"""
        # Сигналы от потока вычислений
        self.compute_worker.signals.started.connect(self.on_computation_started)
        self.compute_worker.signals.progress.connect(self.update_progress)
        self.compute_worker.signals.solution_ready.connect(self.add_solution)
        self.compute_worker.signals.error.connect(self.log_error)
        self.compute_worker.signals.paused.connect(self.on_paused)
        self.compute_worker.signals.resumed.connect(self.on_resumed)
        self.compute_worker.signals.stopped.connect(self.on_stopped)
        self.compute_worker.signals.finished.connect(self.on_finished)

        # Сигналы от потока ввода-вывода
        self.io_worker.signals.loaded.connect(self.load_solutions)
        self.io_worker.signals.saved.connect(self.on_saved)
        self.io_worker.signals.error.connect(self.log_error)

        # Кнопки управления
        self.btn_start.clicked.connect(self.start_computation)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_stop.clicked.connect(self.stop_computation)
        self.btn_clear.clicked.connect(self.clear_solutions)
        self.btn_save.clicked.connect(self.save_solutions)
        self.btn_load.clicked.connect(self.load_file)

        # Кнопки лога
        self.btn_clear_log.clicked.connect(self.clear_log)
        self.btn_save_log.clicked.connect(self.save_log)

        # Кнопки границ параметров
        self.btn_update_bounds.clicked.connect(self.update_param_bounds)
        self.btn_auto_bounds.clicked.connect(self.auto_param_bounds)
        self.btn_reset_bounds.clicked.connect(self.reset_param_bounds)

        # Кнопки решений
        self.btn_export_solution.clicked.connect(self.export_selected_solution)
        self.solutions_table.itemSelectionChanged.connect(self.on_solution_selected)

        # Кнопки визуализации
        self.btn_update_viz.clicked.connect(self.update_visualization)

        # Поиск
        self.search_box.textChanged.connect(self.filter_solutions)
        self.solution_filter.currentIndexChanged.connect(self.filter_solutions)

    def get_current_params(self):
        """Получить текущие значения параметров"""
        a = [spinbox.value() for spinbox in self.param_spinboxes]

        a = np.random.rand(len(a))

        return a

    @Slot()
    def start_computation(self):
        """Запуск вычислений"""
        if self.compute_worker.is_running():
            self.log_message("Вычисления уже выполняются")
            return

        # Получаем выбранную функцию
        func_name = self.func_combo.currentText()
        func = self.available_functions[func_name]

        # Получаем параметры
        params = self.get_current_params()

        # Устанавливаем задачу
        self.compute_worker.set_task(func, params)

        # Запускаем поток
        self.compute_worker.start()

        self.log_message(f"Запущены вычисления функции: {func_name}")
        self.log_message(f"Параметры: {params}")
        self.statusBar().showMessage(f"Выполняется: {func_name}")

    @Slot()
    def on_computation_started(self):
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.stats_timer.start(1000)  # Обновляем статистику каждую секунду

    @Slot(int, str)
    def update_progress(self, percent: int, status: str):
        self.progress_bar.setValue(percent)
        self.status_label.setText(status)

        if percent < 100:
            self.statusBar().showMessage(f"Выполнение: {percent}% - {status}")

    @Slot(dict)
    def add_solution(self, solution_dict):
        """Добавление нового решения"""
        try:
            # Создаем объект Solution
            solution = Solution(
                parameters=solution_dict['parameters'],
                value=solution_dict['value'],
                metadata=solution_dict.get('metadata', {})
            )

            # Добавляем в модель
            self.solution_model.add_solution(solution)

            # Добавляем в таблицу
            self._add_solution_to_table(solution)

            # Обновляем статистику
            self.update_stats()

            # Обновляем параллельные координаты
            if hasattr(self, 'parallel_coords'):
                self.parallel_coords.set_solutions(self.solution_model.solutions[-50:])  # Последние 50 решений

        except Exception as e:
            self.log_error(f"Ошибка добавления решения: {e}")

    def _add_solution_to_table(self, solution: Solution):
        """Добавление решения в таблицу"""
        row = self.solutions_table.rowCount()
        self.solutions_table.insertRow(row)

        # ID
        self.solutions_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))

        # Значение
        value_item = QTableWidgetItem(f"{solution.value:.6f}")
        value_item.setData(Qt.UserRole, solution)
        self.solutions_table.setItem(row, 1, value_item)

        # Параметры
        params_str = ", ".join(f"{p:.4f}" for p in solution.parameters)
        self.solutions_table.setItem(row, 2, QTableWidgetItem(params_str))

        # Время
        time_str = time.strftime("%H:%M:%S", time.localtime(solution.timestamp))
        self.solutions_table.setItem(row, 3, QTableWidgetItem(time_str))

        # Статус
        status = "Новое" if row == len(self.solution_model.solutions) - 1 else "Сохранено"
        self.solutions_table.setItem(row, 4, QTableWidgetItem(status))

        # Действия
        action_widget = QWidget()
        action_layout = QHBoxLayout(action_widget)

        btn_view = QPushButton("👁")
        btn_view.setMaximumWidth(30)
        btn_view.clicked.connect(lambda: self.view_solution(row))

        btn_export = QPushButton("📥")
        btn_export.setMaximumWidth(30)
        btn_export.clicked.connect(lambda: self.export_solution(row))

        action_layout.addWidget(btn_view)
        action_layout.addWidget(btn_export)
        action_layout.setContentsMargins(2, 2, 2, 2)

        self.solutions_table.setCellWidget(row, 5, action_widget)

    def view_solution(self, row):
        """Просмотр выбранного решения"""
        self.solutions_table.selectRow(row)
        self.tab_widget.setCurrentIndex(2)  # Переход на вкладку решений

    def export_solution(self, row):
        """Экспорт выбранного решения"""
        if row < len(self.solution_model.solutions):
            solution = self.solution_model.solutions[row]

            filepath, _ = QFileDialog.getSaveFileName(
                self, "Экспорт решения", "", "JSON (*.json)"
            )

            if filepath:
                data = [{
                    'parameters': solution.parameters,
                    'value': solution.value,
                    'metadata': solution.metadata,
                    'timestamp': solution.timestamp
                }]

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                self.log_message(f"Решение {row + 1} экспортировано в {filepath}")

    @Slot()
    def on_solution_selected(self):
        """Обработка выбора решения в таблице"""
        selected_rows = self.solutions_table.selectedItems()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        if row < len(self.solution_model.solutions):
            solution = self.solution_model.solutions[row]

            self.detail_id.setText(str(row + 1))
            self.detail_value.setText(f"{solution.value:.6f}")
            self.detail_params.setText(", ".join(f"{p:.4f}" for p in solution.parameters))
            self.detail_time.setText(time.strftime("%Y-%m-%d %H:%M:%S",
                                                   time.localtime(solution.timestamp)))

            # Метаданные
            if solution.metadata:
                metadata_text = json.dumps(solution.metadata, indent=2, ensure_ascii=False)
                self.detail_metadata.setText(metadata_text)
            else:
                self.detail_metadata.setText("Нет метаданных")

    @Slot()
    def update_stats(self):
        """Обновление статистики"""
        count = len(self.solution_model.solutions)
        if count > 0:
            best = self.solution_model.get_best(1)[0]
            self.stats_label.setText(
                f"Решений: {count} | Лучшее: {best.value:.6f}"
            )
        else:
            self.stats_label.setText("Решений: 0 | Лучшее: N/A")

    @Slot()
    def toggle_pause(self):
        """Переключение паузы"""
        if self.compute_worker.is_running():
            if not self.compute_worker.is_paused():
                self.compute_worker.pause()
                self.btn_pause.setText("▶ Продолжить")
                self.log_message("Вычисления поставлены на паузу")
            else:
                self.compute_worker.resume()
                self.btn_pause.setText("⏸ Пауза")
                self.log_message("Вычисления возобновлены")

    @Slot()
    def stop_computation(self):
        """Остановка вычислений"""
        if self.compute_worker.is_running():
            self.compute_worker.stop()

    @Slot()
    def on_paused(self):
        self.statusBar().showMessage("Вычисления приостановлены")

    @Slot()
    def on_resumed(self):
        self.statusBar().showMessage("Вычисления возобновлены")

    @Slot()
    def on_stopped(self):
        self.statusBar().showMessage("Вычисления остановлены")
        self.reset_controls()

    @Slot()
    def on_finished(self):
        self.statusBar().showMessage("Вычисления завершены")
        self.reset_controls()

    def reset_controls(self):
        """Сброс элементов управления"""
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setText("⏸ Пауза")
        self.progress_bar.setValue(0)
        self.stats_timer.stop()

    @Slot()
    def clear_solutions(self):
        """Очистка всех решений"""
        reply = QMessageBox.question(
            self, "Подтверждение",
            "Очистить все решения?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.solution_model.clear()
            self.solutions_table.setRowCount(0)
            self.parallel_coords.set_solutions([])
            self.update_stats()
            self.log_message("Все решения очищены")

    @Slot(str)
    def log_error(self, error_msg: str):
        self.log_message(f"❌ ОШИБКА: {error_msg}")
        self.statusBar().showMessage(f"Ошибка: {error_msg}")
        QMessageBox.critical(self, "Ошибка", error_msg)

    def log_message(self, message: str):
        """Добавление сообщения в лог"""
        timestamp = time.strftime("[%H:%M:%S]")
        self.log_text.append(f"{timestamp} {message}")
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    @Slot()
    def clear_log(self):
        """Очистка лога"""
        self.log_text.clear()
        self.log_message("Лог очищен")

    @Slot()
    def save_log(self):
        """Сохранение лога в файл"""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Сохранить лог", "", "Текст (*.txt)"
        )

        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.log_text.toPlainText())
            self.log_message(f"Лог сохранен в {filepath}")

    @Slot()
    def save_solutions(self):
        """Сохранение всех решений"""
        if not self.solution_model.solutions:
            QMessageBox.warning(self, "Предупреждение", "Нет решений для сохранения")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Сохранить решения", "",
            "JSON (*.json);;Binary (*.bin)"
        )

        if filepath:
            binary = filepath.endswith('.bin')
            self.io_worker.save_solutions(
                self.solution_model.get_all(),
                filepath,
                binary
            )

    @Slot()
    def on_saved(self):
        self.log_message("Решения успешно сохранены")
        self.statusBar().showMessage("Решения сохранены")

    @Slot()
    def load_file(self):
        """Загрузка решений из файла"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Загрузить решения", "",
            "JSON (*.json);;Binary (*.bin)"
        )

        if filepath:
            binary = filepath.endswith('.bin')
            self.io_worker.load_solutions(filepath, binary)

    @Slot(list)
    def load_solutions(self, solutions: list):
        """Загрузка решений из файла"""
        self.solution_model.clear()
        self.solutions_table.setRowCount(0)

        for solution in solutions:
            self.solution_model.add_solution(solution)
            self._add_solution_to_table(solution)

        # Обновляем параллельные координаты
        if hasattr(self, 'parallel_coords'):
            self.parallel_coords.set_solutions(self.solution_model.solutions[-50:])

        self.update_stats()
        self.log_message(f"Загружено {len(solutions)} решений")

    @Slot()
    def update_param_bounds(self):
        """Обновление границ параметров из таблицы"""
        self.log_message("Границы параметров обновлены")

    @Slot()
    def auto_param_bounds(self):
        """Автоматическое определение границ параметров"""
        # Здесь можно реализовать логику автоматического определения границ
        self.log_message("Автоматические границы установлены")

    @Slot()
    def reset_param_bounds(self):
        """Сброс границ параметров к значениям по умолчанию"""
        self._update_param_bounds_table()
        self.log_message("Границы параметров сброшены")

    @Slot()
    def export_selected_solution(self):
        """Экспорт выбранного решения"""
        selected_row = self.solutions_table.currentRow()
        if selected_row >= 0:
            self.export_solution(selected_row)

    @Slot()
    def update_visualization(self):
        """Обновление визуализации"""
        self.parallel_coords.update_display()
        self.log_message("Визуализация обновлена")

    @Slot()
    def filter_solutions(self):
        """Фильтрация решений"""
        # Здесь можно реализовать логику фильтрации
        filter_text = self.search_box.text().lower()
        filter_type = self.solution_filter.currentText()

        self.log_message(f"Фильтр применен: {filter_type}, текст: {filter_text}")

    def closeEvent(self, event):
        """Корректное завершение при закрытии"""
        if self.compute_worker.is_running():
            self.compute_worker.stop()
            self.compute_worker.wait()

        if self.io_worker.isRunning():
            self.io_worker.quit()
            self.io_worker.wait()

        event.accept()