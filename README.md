# TBRGS – Traffic-based Route Guidance System

TBRGS is a research project that builds an AI‑driven route guidance system for the Boroondara region (Melbourne, AU) using **SCATS traffic data** and a road network extracted from **OpenStreetMap**.

It combines:

- Deep‑learning traffic prediction models (**LSTM / GRU / CNN**, PyTorch)
- Graph‑based route search algorithms (**GBFS, UCS, A\***)
- An interactive **Streamlit web demo**
- A desktop **PyQt GUI** with an embedded map view

---

## 🌐 Online Demo (Streamlit)


**Live demo:**  
[https://tbrgs-app-us8rbxvwr8wqvnnzzinbcz.streamlit.app/]

The demo allows you to:

- Choose a prediction model (LSTM / GRU / CNN)
- Select origin & destination SCATS sites
- Set weekday & time
- Run different search algorithms (GBFS / UCS / A\*)  
- Visualise the resulting route(s) on an interactive map

---

## 📁 Project Structure

```text
.
├── streamlit_app.py           # Streamlit web UI
├── main_gui.py                # PyQt desktop GUI
├── tbrgs_core.py              # Core pipeline: prediction + routing
├── algorithm.py               # Search algorithms (GBFS, BFS, UCS, A*)
├── Cal_cost_all.py            # Traffic models + travel time assignment
├── map_generator.py           # Folium map generation (with caching)
├── requirements.txt           # Python dependencies for the web demo
├── data/
│   ├── B.osm.pbf              # Road network (OSM extract for Boroondara)
│   ├── mapped_scats_nodes.csv
│   ├── Filtered_SCATS_SiteListing.csv
│   ├── Paths&Node_mapped.csv
│   ├── location_id_mapping.csv
│   └── cleaned_scats_data_with_location_code.csv
└── ML/
    ├── best_lstm_model.pth
    ├── best_gru_model.pth
    └── best_cnn_model.pth
```



---

## ⚙️ Requirements

- Python **3.9+** (tested with 3.11)
- For the **Streamlit** web version:
  - `streamlit`
  - `pandas`
  - `numpy`
  - `torch`
  - `folium`
  - `pyrosm`
- For the **GUI** version (desktop app):
  - `PyQt5`
  - `PyQtWebEngine`

All core dependencies for the web demo are listed in `requirements.txt`.

---

## 🚀 Getting Started (Local)

### 1. Clone the repository

```bash
git clone https://github.com/Exist-Liang/tbrgs-streamlit.git
cd tbrgs-streamlit
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies for the Streamlit demo

```bash
pip install -r requirements.txt
```

### 4. Install extra dependencies for the GUI

```bash
pip install PyQt5 PyQtWebEngine
```

Make sure you have placed the required data and model files under `data/` and `ML/` before running.

---

## 🌐 Running the Streamlit Web Demo (Locally)

```bash
streamlit run streamlit_app.py
```

Then open the URL printed in the terminal (usually `http://localhost:8501`).

The web interface lets you:

1. Select:
   - **Model**: LSTM / GRU / CNN  
   - **Search algorithm**: GBFS / UCS / A\*
   - **Origin & destination SCATS** (with human‑readable descriptions)
   - **Weekday & time**
2. Click **“Run Prediction”** to:
   - Run the selected deep‑learning model to predict link travel times  
   - Build a weighted road network  
   - Search for one or multiple routes according to the chosen algorithm  
   - Display the top routes and their estimated travel times  
   - Render them on an interactive Folium map

> First run may take longer because the models, historical data and OSM network are loaded and cached.  
> Subsequent runs in the same session are significantly faster.

---

## 🖥️ Running the Desktop GUI

The desktop GUI provides a similar workflow but as a standalone application, with an embedded browser view for the map.

```bash
python main_gui.py
```

GUI features:

- Drop‑downs for:
  - Model (LSTM / GRU / CNN)
  - Algorithm (GBFS / UCS / A\*)
  - Origin & destination SCATS
- Button groups for:
  - Hour & minute selection (00 / 15 / 30 / 45)
  - Weekday selection (Mon–Sun)
- “Random” button to sample a random origin/destination/time
- Text area listing:
  - Chosen parameters
  - Top‑k routes and their estimated travel times
- Embedded map view (via `QWebEngineView`) showing:
  - Base SCATS network
  - Highlighted routes with start/end markers

The GUI internally calls the same core logic as the Streamlit app (`run_prediction_logic` in `tbrgs_core.py`) and uses `map_generator.py` to render the map.

---

## 📚 Core Logic Overview

1. **Traffic prediction** (`Cal_cost_all.py`)
   - Loads historical SCATS data and a mapping from SCATS site to internal location ID
   - Builds input sequences (volume, weekday, hour, location ID)
   - Uses the selected deep‑learning model (LSTM / GRU / CNN) to predict traffic flow
   - Converts flow to speed & travel time for each edge

2. **Graph construction** (`tbrgs_core.py`)
   - Builds a directed graph from `Paths&Node_mapped.csv`
   - Each edge is weighted by predicted `Travel_Time_Min`

3. **Route search** (`algorithm.py`)
   - Supports:
     - **GBFS** – Greedy Best‑First Search
     - **UCS** – Uniform Cost Search
     - **A\*** – A* with heuristic
   - Returns top‑k paths and their estimated travel times

4. **Visualisation** (`map_generator.py`)
   - Uses `pyrosm` + `folium` to:
     - Load road network from `B.osm.pbf`
     - Map path nodes to OSM coordinates
     - Draw polylines and markers on an interactive map
