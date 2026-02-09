from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QListWidgetItem,
    QGroupBox, QLineEdit, QLabel, QProgressBar,
    QTabWidget, QTextEdit, QFileDialog, QMessageBox,
    QComboBox, QSpinBox, QDoubleSpinBox, QFormLayout,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QCheckBox
)
from PySide6.QtCore import Qt, Slot, QTimer
import json

from core.signals import ComputeSignals, IOSignals
from core.solution_model import Solution, SolutionModel
from core.workers.compute_worker import ComputeWorker
from core.workers.io_worker import IOWorker
from core.compute_context import example_function_1, monte_carlo_simulation


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CZ Gate Optimization Strategies")


        # Модель данных
        self.solution_model = SolutionModel()

        # Потоки
        self.compute_worker = ComputeWorker()
        self.io_worker = IOWorker()

        # Список доступных функций
        self.available_functions = {
            "Пример функции 1 (пошаговая)": example_function_1,
            "Монте-Карло симуляция": monte_carlo_simulation,
        }

        # Таймер для обновления статистики
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats)

        # Инициализация UI и подключение сигналов
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Панель выбора функции
        func_group = QGroupBox("Параметры вычислений")
        func_layout = QVBoxLayout()

        # Выбор функции
        func_select_layout = QHBoxLayout()
        func_select_layout.addWidget(QLabel("Функция:"))
        self.func_combo = QComboBox()
        self.func_combo.addItems(self.available_functions.keys())
        func_select_layout.addWidget(self.func_combo)
        func_layout.addLayout(func_select_layout)

        # Параметры функции
        param_layout = QFormLayout()
        self.param_inputs = []

        # Количество параметров
        self.param_count_spin = QSpinBox()
        self.param_count_spin.setRange(1, 10)
        self.param_count_spin.setValue(2)
        self.param_count_spin.valueChanged.connect(self._update_param_controls)
        param_layout.addRow("Количество параметров:", self.param_count_spin)

        # Динамические поля для параметров
        self.param_widget = QWidget()
        self.param_widget_layout = QFormLayout(self.param_widget)
        param_layout.addRow("Значения параметров:", self.param_widget)

        func_layout.addLayout(param_layout)
        func_group.setLayout(func_layout)
        main_layout.addWidget(func_group)

        # Создаем начальные элементы управления
        self._update_param_controls()

        # Панель управления вычислениями
        control_group = QGroupBox("Управление вычислениями")
        control_layout = QVBoxLayout()

        # Кнопки управления
        button_layout = QHBoxLayout()

        self.btn_start = QPushButton("Запуск")
        self.btn_pause = QPushButton("Пауза")
        self.btn_stop = QPushButton("Стоп")
        self.btn_clear = QPushButton("Очистить решения")

        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)

        button_layout.addWidget(self.btn_start)
        button_layout.addWidget(self.btn_pause)
        button_layout.addWidget(self.btn_stop)
        button_layout.addWidget(self.btn_clear)

        control_layout.addLayout(button_layout)

        # Статистика
        stats_layout = QHBoxLayout()
        self.stats_label = QLabel("Решений: 0 | Лучшее: N/A")
        stats_layout.addWidget(self.stats_label)
        control_layout.addLayout(stats_layout)

        # Прогресс
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Готов")
        control_layout.addWidget(self.progress_bar)
        control_layout.addWidget(self.status_label)

        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)

        # Табы для отображения
        tabs = QTabWidget()

        # Таблица решений
        self.solutions_table = QTableWidget()
        self.solutions_table.setColumnCount(5)
        self.solutions_table.setHorizontalHeaderLabels([
            "ID", "Значение", "Параметры", "Время", "Метаданные"
        ])
        self.solutions_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tabs.addTab(self.solutions_table, "Все решения")

        # Лучшие решения
        self.best_solutions_table = QTableWidget()
        self.best_solutions_table.setColumnCount(4)
        self.best_solutions_table.setHorizontalHeaderLabels([
            "Ранг", "Значение", "Параметры", "Улучшение"
        ])
        tabs.addTab(self.best_solutions_table, "Лучшие решения")

        # Лог
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        tabs.addTab(self.log_text, "Лог")

        main_layout.addWidget(tabs)

        # Панель сохранения/загрузки
        io_layout = QHBoxLayout()

        self.btn_save = QPushButton("Сохранить все")
        self.btn_load = QPushButton("Загрузить")
        self.btn_export_best = QPushButton("Экспорт лучших")
        self.btn_export_selected = QPushButton("Экспорт выбранного")

        io_layout.addWidget(self.btn_save)
        io_layout.addWidget(self.btn_load)
        io_layout.addWidget(self.btn_export_best)
        io_layout.addWidget(self.btn_export_selected)

        main_layout.addLayout(io_layout)

    def _update_param_controls(self):
        """Обновление элементов управления параметрами"""
        # Очистка старых элементов
        while self.param_widget_layout.rowCount() > 0:
            self.param_widget_layout.removeRow(0)

        # Создание новых элементов
        param_count = self.param_count_spin.value()
        self.param_spinboxes = []

        for i in range(param_count):
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-100, 100)
            spinbox.setValue(1.0 if i % 2 == 0 else -1.0)
            spinbox.setDecimals(6)
            self.param_widget_layout.addRow(f"Параметр {i + 1}:", spinbox)
            self.param_spinboxes.append(spinbox)

    def _connect_signals(self):
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
        self.io_worker.signals.saved.connect(lambda: self.log_message("Сохранено"))
        self.io_worker.signals.error.connect(self.log_error)

        # Кнопки
        self.btn_start.clicked.connect(self.start_computation)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_stop.clicked.connect(self.stop_computation)
        self.btn_clear.clicked.connect(self.clear_solutions)

        self.btn_save.clicked.connect(self.save_all)
        self.btn_load.clicked.connect(self.load_file)
        self.btn_export_best.clicked.connect(self.export_best)
        self.btn_export_selected.clicked.connect(self.export_selected)

    def get_current_params(self):
        """Получить текущие значения параметров"""
        return [spinbox.value() for spinbox in self.param_spinboxes]

    @Slot()
    def start_computation(self):
        """Запуск вычислений"""
        if self.compute_worker.isRunning():
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

    @Slot(dict)
    def add_solution(self, solution_dict):
        """Добавление нового решения из потока вычислений"""
        try:
            # Создаем объект Solution из словаря
            solution = Solution(
                parameters=solution_dict['parameters'],
                value=solution_dict['value'],
                metadata=solution_dict.get('metadata', {})
            )

            # Добавляем в модель
            self.solution_model.add_solution(solution)

            # Добавляем в таблицу
            row = self.solutions_table.rowCount()
            self.solutions_table.insertRow(row)

            self.solutions_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            self.solutions_table.setItem(row, 1, QTableWidgetItem(f"{solution.value:.6f}"))
            self.solutions_table.setItem(row, 2, QTableWidgetItem(
                ", ".join(f"{p:.4f}" for p in solution.parameters)
            ))
            self.solutions_table.setItem(row, 3, QTableWidgetItem(
                f"{solution.timestamp:.1f}"
            ))

            # Метаданные как JSON
            metadata_str = json.dumps(solution.metadata) if solution.metadata else ""
            self.solutions_table.setItem(row, 4, QTableWidgetItem(metadata_str))

            # Прокручиваем к новому решению
            self.solutions_table.scrollToBottom()

            # Обновляем статистику
            self.update_stats()

        except Exception as e:
            self.log_error(f"Ошибка добавления решения: {e}")

    @Slot()
    def update_stats(self):
        """Обновление статистики"""
        count = len(self.solution_model.solutions)
        if count > 0:
            best = self.solution_model.get_best(1)[0]
            self.stats_label.setText(
                f"Решений: {count} | Лучшее: {best.value:.6f}"
            )

            # Обновляем таблицу лучших решений
            self.update_best_solutions_table()
        else:
            self.stats_label.setText("Решений: 0 | Лучшее: N/A")

    def update_best_solutions_table(self):
        """Обновление таблицы лучших решений"""
        best_solutions = self.solution_model.get_best(10)

        self.best_solutions_table.setRowCount(len(best_solutions))

        for i, solution in enumerate(best_solutions):
            self.best_solutions_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.best_solutions_table.setItem(i, 1, QTableWidgetItem(f"{solution.value:.6f}"))
            self.best_solutions_table.setItem(i, 2, QTableWidgetItem(
                ", ".join(f"{p:.4f}" for p in solution.parameters)
            ))

            # Вычисление улучшения относительно предыдущего
            improvement = ""
            if i > 0:
                prev_value = best_solutions[i - 1].value
                improvement_percent = ((prev_value - solution.value) / abs(prev_value)) * 100
                improvement = f"{improvement_percent:.2f}%"

            self.best_solutions_table.setItem(i, 3, QTableWidgetItem(improvement))

    @Slot()
    def toggle_pause(self):
        """Переключение паузы"""
        if self.compute_worker.is_running():
            if not self.compute_worker.is_paused():
                self.compute_worker.pause()
                self.btn_pause.setText("Продолжить")
                self.log_message("Вычисления поставлены на паузу")
            else:
                self.compute_worker.resume()
                self.btn_pause.setText("Пауза")
                self.log_message("Вычисления возобновлены")

    @Slot()
    def stop_computation(self):
        """Остановка вычислений"""
        if self.compute_worker.isRunning():
            self.compute_worker.stop()

    @Slot()
    def on_paused(self):
        self.log_message("Вычисления приостановлены")

    @Slot()
    def on_resumed(self):
        self.log_message("Вычисления возобновлены")

    @Slot()
    def on_stopped(self):
        self.log_message("Вычисления остановлены")
        self.reset_controls()

    @Slot()
    def on_finished(self):
        self.log_message("Вычисления завершены")
        self.reset_controls()

    def reset_controls(self):
        """Сброс элементов управления после завершения"""
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setText("Пауза")
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
            self.best_solutions_table.setRowCount(0)
            self.update_stats()
            self.log_message("Все решения очищены")

    @Slot(str)
    def log_error(self, error_msg: str):
        self.log_message(f"ОШИБКА: {error_msg}")
        QMessageBox.critical(self, "Ошибка", error_msg)

    def log_message(self, message: str):
        self.log_text.append(message)

    @Slot()
    def save_all(self):
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

            row = self.solutions_table.rowCount()
            self.solutions_table.insertRow(row)

            self.solutions_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            self.solutions_table.setItem(row, 1, QTableWidgetItem(f"{solution.value:.6f}"))
            self.solutions_table.setItem(row, 2, QTableWidgetItem(
                ", ".join(f"{p:.4f}" for p in solution.parameters)
            ))
            self.solutions_table.setItem(row, 3, QTableWidgetItem(
                f"{solution.timestamp:.1f}"
            ))

            metadata_str = json.dumps(solution.metadata) if solution.metadata else ""
            self.solutions_table.setItem(row, 4, QTableWidgetItem(metadata_str))

        self.update_stats()
        self.log_message(f"Загружено {len(solutions)} решений")

    @Slot()
    def export_best(self):
        """Экспорт лучших решений"""
        best_solutions = self.solution_model.get_best(10)

        if not best_solutions:
            QMessageBox.warning(self, "Предупреждение", "Нет решений для экспорта")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Экспорт лучших решений", "", "JSON (*.json)"
        )

        if filepath:
            data = [
                {
                    'parameters': sol.parameters,
                    'value': sol.value,
                    'metadata': sol.metadata,
                    'timestamp': sol.timestamp
                }
                for sol in best_solutions
            ]

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            self.log_message(f"Экспортировано {len(best_solutions)} лучших решений")

    @Slot()
    def export_selected(self):
        """Экспорт выбранного решения"""
        selected_row = self.solutions_table.currentRow()

        if selected_row < 0:
            QMessageBox.warning(self, "Предупреждение", "Выберите решение для экспорта")
            return

        if selected_row < len(self.solution_model.solutions):
            solution = self.solution_model.solutions[selected_row]

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

                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)

                self.log_message("Решение экспортировано")

    def closeEvent(self, event):
        """Корректное завершение при закрытии"""
        if self.compute_worker.isRunning():
            self.compute_worker.stop()
            self.compute_worker.wait()

        if self.io_worker.isRunning():
            self.io_worker.quit()
            self.io_worker.wait()

        self.stats_timer.stop()
        event.accept()