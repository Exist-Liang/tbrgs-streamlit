import sys
import os
import random
import json

import pandas as pd
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QLabel,
    QMessageBox,
    QLineEdit,
    QTextEdit,
    QSplitter,
    QGridLayout,
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, Qt

from map_generator import generate_map, generate_base_map
from tbrgs_core import run_prediction_logic


CONFIG_PATH = "gui_config.json"


def load_scats_and_descriptions():
    """
    Load SCATS IDs and their descriptions from CSV files.

    - data/mapped_scats_nodes.csv (column: SCATS_ID)
    - data/Filtered_SCATS_SiteListing.csv (columns: SCAT ID, Location Description)
    """
    scats_file = "data/mapped_scats_nodes.csv"
    desc_file = "data/Filtered_SCATS_SiteListing.csv"

    if not os.path.exists(scats_file):
        raise FileNotFoundError(f"Required file not found: {scats_file}")

    scats_df = pd.read_csv(scats_file)
    scats_ids = (
        scats_df["SCATS_ID"]
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
        .tolist()
    )

    desc_map = {}
    if os.path.exists(desc_file):
        desc_df = pd.read_csv(desc_file)
        desc_df = desc_df.dropna(subset=["SCAT ID", "Location Description"])
        desc_map = {
            int(row["SCAT ID"]): row["Location Description"]
            for _, row in desc_df.iterrows()
        }

    return scats_ids, desc_map


class TBRGSApp(QWidget):
    """
    PyQt5 GUI wrapper around run_prediction_logic + map_generator.

    左侧：参数输入（模型、算法、起终点、时间、星期）
    右侧：地图（QWebEngineView）
    下方：文本输出路由信息
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("TBRGS – Traffic-based Route Guidance System (GUI)")
        self.resize(1200, 800)

        # --- data ---
        self.scats_ids, self.desc_map = load_scats_and_descriptions()

        # --- config (optional) ---
        self.config = self.load_config()

        # Selected values (for buttons)
        self.selected_minute = self.config.get("minute", "30")
        self.selected_weekday = int(self.config.get("weekday", 4))

        # Build UI
        self.init_ui()

        # Ensure we have a base map ready
        self.ensure_base_map()

    # ------------------------ config ------------------------

    def load_config(self):
        """
        Try to load previous GUI config from gui_config.json
        """
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data
            except Exception:
                pass

        # Default config
        return {
            "model": "LSTM",
            "algorithm": "astar",
            "origin": self.scats_ids[0] if self.scats_ids else None,
            "destination": self.scats_ids[1] if len(self.scats_ids) > 1 else (self.scats_ids[0] if self.scats_ids else None),
            "hour": "09",
            "minute": "30",
            "weekday": 4,  # Friday
        }

    def save_config(self):
        """
        Save current GUI selections to gui_config.json
        """
        cfg = {
            "model": self.model_cb.currentText(),
            "algorithm": self.algo_cb.currentText(),
            "origin": self.origin_input.currentData(),
            "destination": self.dest_input.currentData(),
            "hour": self.hour_input.text().zfill(2),
            "minute": self.selected_minute,
            "weekday": self.selected_weekday,
        }
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
        except Exception as e:
            print("Failed to save config:", e)

    # ------------------------ UI ------------------------

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel: controls + log
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(8)

        # --- Model & Algorithm ---
        grid = QGridLayout()
        row = 0

        grid.addWidget(QLabel("Model:"), row, 0)
        self.model_cb = QComboBox()
        self.model_cb.addItems(["LSTM", "GRU", "CNN"])
        default_model = self.config.get("model", "LSTM")
        idx = self.model_cb.findText(default_model)
        if idx >= 0:
            self.model_cb.setCurrentIndex(idx)
        grid.addWidget(self.model_cb, row, 1)
        row += 1

        grid.addWidget(QLabel("Algorithm:"), row, 0)
        self.algo_cb = QComboBox()
        # DFS 已移除，只保留性能可接受的算法
        self.algo_cb.addItems(["gbfs", "ucs", "astar"])
        default_alg = self.config.get("algorithm", "astar")
        idx = self.algo_cb.findText(default_alg)
        if idx >= 0:
            self.algo_cb.setCurrentIndex(idx)
        grid.addWidget(self.algo_cb, row, 1)
        row += 1

        # --- Origin / Destination ---
        def format_scats(sid: int) -> str:
            return f"{sid} – {self.desc_map.get(sid, 'Unknown')}"

        grid.addWidget(QLabel("Origin SCATS:"), row, 0)
        self.origin_input = QComboBox()
        for sid in self.scats_ids:
            self.origin_input.addItem(format_scats(sid), sid)
        origin_default = self.config.get("origin")
        if origin_default in self.scats_ids:
            idx_o = self.origin_input.findData(origin_default)
            if idx_o >= 0:
                self.origin_input.setCurrentIndex(idx_o)
        grid.addWidget(self.origin_input, row, 1)
        row += 1

        grid.addWidget(QLabel("Destination SCATS:"), row, 0)
        self.dest_input = QComboBox()
        for sid in self.scats_ids:
            self.dest_input.addItem(format_scats(sid), sid)
        dest_default = self.config.get("destination")
        if dest_default in self.scats_ids:
            idx_d = self.dest_input.findData(dest_default)
            if idx_d >= 0:
                self.dest_input.setCurrentIndex(idx_d)
        grid.addWidget(self.dest_input, row, 1)
        row += 1

        left_layout.addLayout(grid)

        # --- Time selection ---
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Hour:"))
        self.hour_input = QLineEdit(self.config.get("hour", "09").zfill(2))
        self.hour_input.setFixedWidth(40)
        self.hour_input.setMaxLength(2)
        self.hour_input.setPlaceholderText("00-23")
        time_layout.addWidget(self.hour_input)

        time_layout.addWidget(QLabel("Minute:"))
        self.minute_buttons = []
        for m in ["00", "15", "30", "45"]:
            btn = QPushButton(m)
            btn.setCheckable(True)
            btn.setFixedWidth(40)
            btn.clicked.connect(self.update_minute_selection)
            btn.setChecked(m == self.selected_minute)
            self.minute_buttons.append(btn)
            time_layout.addWidget(btn)

        left_layout.addLayout(time_layout)

        # --- Weekday selection ---
        weekday_layout = QHBoxLayout()
        weekday_layout.addWidget(QLabel("Weekday:"))

        self.weekday_buttons = []
        weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for idx, label in enumerate(weekday_labels):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedWidth(45)
            btn.clicked.connect(lambda checked, i=idx: self.set_weekday(i))
            btn.setChecked(idx == self.selected_weekday)
            self.weekday_buttons.append(btn)
            weekday_layout.addWidget(btn)

        left_layout.addLayout(weekday_layout)

        # --- Buttons (Run / Random) ---
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.run_prediction)
        btn_row.addWidget(self.run_btn)

        self.random_btn = QPushButton("Random")
        self.random_btn.clicked.connect(self.randomize_inputs)
        btn_row.addWidget(self.random_btn)

        left_layout.addLayout(btn_row)

        # --- Text output area ---
        left_layout.addWidget(QLabel("Route Output:"))
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMinimumHeight(250)
        left_layout.addWidget(self.output_text)

        # Left widget into splitter
        splitter.addWidget(left_widget)

        # Right panel: map view
        self.web_view = QWebEngineView()
        splitter.addWidget(self.web_view)

        splitter.setSizes([400, 800])

    # ------------------------ interactions ------------------------

    def ensure_base_map(self):
        """Ensure there is at least a base map to show."""
        try:
            os.makedirs("data", exist_ok=True)
            output_html = "data/output_map.html"
            if not os.path.exists(output_html):
                generate_base_map(
                    output_html=output_html,
                    pbf_file="data/B.osm.pbf",
                    node_file="data/mapped_scats_nodes.csv",
                )
            self.load_map(output_html)
        except Exception as e:
            print("Failed to generate base map:", e)

    def load_map(self, html_file: str):
        if os.path.exists(html_file):
            url = QUrl.fromLocalFile(os.path.abspath(html_file))
            self.web_view.setUrl(url)
        else:
            self.output_text.append(f"Map file not found: {html_file}")

    def update_minute_selection(self):
        sender = self.sender()
        if not isinstance(sender, QPushButton):
            return
        minute = sender.text()
        self.selected_minute = minute
        for btn in self.minute_buttons:
            btn.setChecked(btn is sender)

    def set_weekday(self, idx: int):
        self.selected_weekday = idx
        for i, btn in enumerate(self.weekday_buttons):
            btn.setChecked(i == idx)

    def randomize_inputs(self):
        """Randomly choose origin/destination/time/weekday."""
        if len(self.scats_ids) >= 2:
            origin, dest = random.sample(self.scats_ids, 2)
        elif len(self.scats_ids) == 1:
            origin = dest = self.scats_ids[0]
        else:
            QMessageBox.warning(self, "No SCATS data", "SCATS list is empty.")
            return

        # set combos
        for i in range(self.origin_input.count()):
            if self.origin_input.itemData(i) == origin:
                self.origin_input.setCurrentIndex(i)
                break
        for i in range(self.dest_input.count()):
            if self.dest_input.itemData(i) == dest:
                self.dest_input.setCurrentIndex(i)
                break

        # hour & minute
        hour = random.randint(0, 23)
        self.hour_input.setText(str(hour).zfill(2))
        minute = random.choice(["00", "15", "30", "45"])
        self.selected_minute = minute
        for btn in self.minute_buttons:
            btn.setChecked(btn.text() == minute)

        # weekday
        weekday_idx = random.randint(0, 6)
        self.set_weekday(weekday_idx)

    def run_prediction(self):
        origin = self.origin_input.currentData()
        destination = self.dest_input.currentData()

        if origin is None or destination is None:
            QMessageBox.warning(self, "Invalid Input", "Please select valid origin and destination!")
            return

        if origin == destination:
            QMessageBox.warning(self, "Invalid Input", "Origin and destination cannot be the same!")
            return

        model = self.model_cb.currentText()
        algorithm = self.algo_cb.currentText()
        hour_str = self.hour_input.text().zfill(2)
        minute_str = self.selected_minute

        # Basic validation for hour
        try:
            hour_int = int(hour_str)
            if not (0 <= hour_int <= 23):
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Invalid Time", "Hour must be an integer between 0 and 23.")
            return

        time_str = f"{hour_str}:{minute_str}"
        weekday_idx = self.selected_weekday

        self.output_text.clear()
        self.output_text.append(
            f"Running prediction:\n"
            f"  Model: {model}\n"
            f"  Algorithm: {algorithm}\n"
            f"  Origin: {origin}\n"
            f"  Destination: {destination}\n"
            f"  Weekday: {weekday_idx}\n"
            f"  Time: {time_str}\n"
        )

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.run_btn.setEnabled(False)
        self.random_btn.setEnabled(False)

        try:
            results = run_prediction_logic(
                model=model,
                weekday=weekday_idx,
                time_str=time_str,
                origin=int(origin),
                destination=int(destination),
                algorithm=algorithm,
                top_k=5,
            )

            if not results:
                self.output_text.append("\n❌ No path found.")
                return

            self.output_text.append("✅ Routes found:\n")
            for i, item in enumerate(results, start=1):
                path_str = " → ".join(map(str, item["path"]))
                self.output_text.append(
                    f"Route {i}: {path_str}\n"
                    f"  Estimated time: {item['cost']:.2f} min\n"
                )

            # Generate and load map
            path_nodes_list = [r["path"] for r in results]
            path_cost_list = [r["cost"] for r in results]
            output_html = "data/output_map.html"

            try:
                generate_map(
                    path_nodes_list=path_nodes_list,
                    path_cost_list=path_cost_list,
                    pbf_file="data/B.osm.pbf",
                    path_file="data/Paths&Node_mapped.csv",
                    node_file="data/mapped_scats_nodes.csv",
                    output_html=output_html,
                )
                self.load_map(output_html)
            except Exception as e:
                print("Map generation failed:", e)
                self.output_text.append(f"\n⚠ Map generation failed: {e}")

            # 保存当前配置
            self.save_config()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed:\n{e}")
            print("Prediction failed:", e)
        finally:
            QApplication.restoreOverrideCursor()
            self.run_btn.setEnabled(True)
            self.random_btn.setEnabled(True)


if __name__ == "__main__":
    import traceback

    try:
        app = QApplication(sys.argv)
        win = TBRGSApp()
        win.show()
        sys.exit(app.exec_())
    except Exception as e:
        print("Application exited with error:", e)
        traceback.print_exc()
