import json
import sys
import os
import random
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QMessageBox, QLineEdit, QTextEdit,
    QScrollArea, QSizePolicy, QSplitter, QGridLayout
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, QDateTime, Qt
from map_generator import generate_map, generate_base_map
from tbrgs_core import run_prediction_logic

class TBRGSApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TBRGS Path Search Visualization")

        self.config = {
            "model": "LSTM",
            "algorithm": "astar",
            "origin": None,
            "destination": None,
            "hour": "09",
            "minute": "30",
            "weekday": 4
        }
        try:
            if os.path.exists("GUIconfig.json"):
                with open("GUIconfig.json", "r") as f:
                    user_config = json.load(f)
                    self.config.update(user_config)
        except Exception as e:
            print("\u26a0\ufe0f Failed to load config.json:", e)

        self.setGeometry(100, 100, 1200, 600)

        try:
            generate_base_map(
                output_html="data/output_map.html",
                pbf_file="data/B.osm.pbf",
                node_file="data/mapped_scats_nodes.csv"
            )
        except Exception as e:
            print("Failed to initialize map:", e)

        try:
            scats_df = pd.read_csv("data/mapped_scats_nodes.csv")
            scats_ids = sorted(scats_df["SCATS_ID"].dropna().astype(int).unique().tolist())
        except Exception as e:
            print("\u26a0\ufe0f Failed to load node list:", e)
            scats_ids = []

        try:
            desc_df = pd.read_csv("data/Filtered_SCATS_SiteListing.csv")
            desc_df = desc_df.dropna(subset=["SCAT ID", "Location Description"])
            desc_map = {
                int(row["SCAT ID"]): row["Location Description"]
                for _, row in desc_df.iterrows()
            }
        except Exception as e:
            print("\u26a0\ufe0f Failed to load description file:", e)
            desc_map = {}

        self.scats_ids = scats_ids
        self.desc_map = desc_map

        self.model_cb = QComboBox()
        self.model_cb.addItems(["LSTM", "GRU", "CNN"])
        self.model_cb.setCurrentText(self.config["model"])

        self.algo_cb = QComboBox()
        self.algo_cb.addItems(["gbfs", "dfs", "ucs", "astar"])
        self.algo_cb.setCurrentText(self.config["algorithm"])

        self.origin_input = QComboBox()
        self.dest_input = QComboBox()
        for sid in scats_ids:
            desc = desc_map.get(sid, "Unknown")
            label = f"{sid} - {desc}"
            self.origin_input.addItem(label, sid)
            self.dest_input.addItem(label, sid)

        if self.config["origin"] is not None:
            for i in range(self.origin_input.count()):
                if self.origin_input.itemData(i) == self.config["origin"]:
                    self.origin_input.setCurrentIndex(i)
                    break

        if self.config["destination"] is not None:
            for i in range(self.dest_input.count()):
                if self.dest_input.itemData(i) == self.config["destination"]:
                    self.dest_input.setCurrentIndex(i)
                    break

        self.hour_input = QLineEdit(self.config["hour"].zfill(2))

        self.minute_buttons = [QPushButton(m) for m in ["00", "15", "30", "45"]]
        for btn in self.minute_buttons:
            btn.setCheckable(True)
            btn.clicked.connect(self.update_minute_selection)
        self.selected_minute = self.config["minute"]
        for btn in self.minute_buttons:
            btn.setChecked(btn.text() == self.selected_minute)

        self.weekday_buttons = []
        weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        self.selected_weekday = self.config["weekday"]
        for i, label in enumerate(weekday_labels):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, idx=i: self.set_weekday(idx))
            btn.setChecked(i == self.selected_weekday)
            self.weekday_buttons.append(btn)

        run_btn = QPushButton("Run")
        run_btn.setFixedHeight(30)
        run_btn.clicked.connect(self.run_prediction)

        random_btn = QPushButton("Random")
        random_btn.setFixedHeight(30)
        random_btn.clicked.connect(self.randomize_inputs)

        form_layout = QVBoxLayout()
        form_layout.setSpacing(8)
        form_layout.addWidget(QLabel("Model"))
        form_layout.addWidget(self.model_cb)
        form_layout.addWidget(QLabel("Origin SCATS"))
        form_layout.addWidget(self.origin_input)
        form_layout.addWidget(QLabel("Destination SCATS"))
        form_layout.addWidget(self.dest_input)
        form_layout.addWidget(QLabel("Algorithm"))
        form_layout.addWidget(self.algo_cb)

        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Hour:"))
        time_layout.addWidget(self.hour_input)
        time_layout.addStretch()
        form_layout.addWidget(QLabel("Time (HH:MM)"))
        form_layout.addLayout(time_layout)

        minute_layout = QHBoxLayout()
        for btn in self.minute_buttons:
            minute_layout.addWidget(btn)
        form_layout.addLayout(minute_layout)

        form_layout.addWidget(QLabel("Weekday"))
        weekday_layout = QGridLayout()
        for i, btn in enumerate(self.weekday_buttons):
            weekday_layout.addWidget(btn, i // 4, i % 4)
        form_layout.addLayout(weekday_layout)

        form_layout.addWidget(random_btn)
        form_layout.addWidget(run_btn)
        form_layout.addStretch(1)

        form_widget = QWidget()
        form_widget.setLayout(form_layout)

        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        self.result_box.setPlaceholderText("Path results will be shown here")
        self.result_box.setMinimumHeight(100)

        left_splitter = QSplitter(Qt.Vertical)
        left_splitter.addWidget(form_widget)
        left_splitter.addWidget(self.result_box)
        left_splitter.setSizes([300, 100])

        self.web = QWebEngineView()
        self.web.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.load_map("data/output_map.html")

        layout = QHBoxLayout()
        layout.addWidget(left_splitter, 2)
        layout.addWidget(self.web, 8)
        self.setLayout(layout)

    def load_map(self, html_file):
        timestamp = QDateTime.currentDateTime().toMSecsSinceEpoch()
        url = QUrl.fromLocalFile(os.path.abspath(html_file))
        url.setQuery(f"v={timestamp}")
        self.web.setUrl(url)

    def update_minute_selection(self):
        for btn in self.minute_buttons:
            if btn.isChecked():
                self.selected_minute = btn.text()
            else:
                btn.setChecked(False)

    def set_weekday(self, idx):
        self.selected_weekday = idx
        for i, btn in enumerate(self.weekday_buttons):
            btn.setChecked(i == idx)

    def randomize_inputs(self):
        if len(self.scats_ids) < 2:
            return
        origin, dest = random.sample(self.scats_ids, 2)
        for i in range(self.origin_input.count()):
            if self.origin_input.itemData(i) == origin:
                self.origin_input.setCurrentIndex(i)
            if self.dest_input.itemData(i) == dest:
                self.dest_input.setCurrentIndex(i)

        self.hour_input.setText(str(random.randint(0, 23)).zfill(2))
        minute = random.choice(["00", "15", "30", "45"])
        for btn in self.minute_buttons:
            btn.setChecked(btn.text() == minute)
        self.selected_minute = minute
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
        hour = self.hour_input.text().zfill(2)
        minute = self.selected_minute
        time_str = f"{hour}:{minute}"
        weekday = self.selected_weekday

        try:
            results = run_prediction_logic(
                model=model,
                weekday=weekday,
                time_str=time_str,
                origin=origin,
                destination=destination,
                algorithm=algorithm,
                top_k=5
            )
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", str(e))
            return

        if not results:
            self.result_box.setText("\u274c No path found")
            return

        text_output = []
        all_paths = []
        all_costs = []
        for i, route in enumerate(results):
            text_output.append(f"Route {i+1}: {' -> '.join(map(str, route['path']))}  Cost: {route['cost']} min")
            all_paths.append(route['path'])
            all_costs.append(route['cost'])

        self.result_box.setText("\n".join(text_output))

        try:
            generate_map(
                path_nodes_list=all_paths,
                path_cost_list=all_costs,
                pbf_file="data/B.osm.pbf",
                path_file="data/Paths&Node_mapped.csv",
                node_file="data/mapped_scats_nodes.csv",
                output_html="data/output_map.html"
            )
            self.load_map("data/output_map.html")
        except Exception as e:
            print("Map generation failed:", e)

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