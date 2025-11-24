import os
import random

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from map_generator import generate_map, generate_base_map
from tbrgs_core import run_prediction_logic


# -----------------------------
# Helpers to load data & config
# -----------------------------


@st.cache_data
def load_scats_and_descriptions():
    """
    Load SCATS IDs and their human-readable descriptions.

    Expects:
      - data/mapped_scats_nodes.csv  with column "SCATS_ID"
      - data/Filtered_SCATS_SiteListing.csv with columns "SCAT ID", "Location Description"
    """
    scats_df = pd.read_csv("data/mapped_scats_nodes.csv")
    scats_ids = (
        scats_df["SCATS_ID"]
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
        .tolist()
    )

    try:
        desc_df = pd.read_csv("data/Filtered_SCATS_SiteListing.csv")
        desc_df = desc_df.dropna(subset=["SCAT ID", "Location Description"])
        desc_map = {
            int(row["SCAT ID"]): row["Location Description"]
            for _, row in desc_df.iterrows()
        }
    except FileNotFoundError:
        desc_map = {}

    return scats_ids, desc_map


@st.cache_resource
def ensure_base_map():
    """
    Generate a base map (SCATS nodes only) once per session
    so that the right-hand side has something to show
    even before the first prediction.
    """
    os.makedirs("data", exist_ok=True)
    output_html = "data/output_map.html"
    if not os.path.exists(output_html):
        generate_base_map(
            output_html=output_html,
            pbf_file="data/B.osm.pbf",
            node_file="data/mapped_scats_nodes.csv",
        )
    return output_html


def load_map_html(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# -----------------------------
# Streamlit page configuration
# -----------------------------

st.set_page_config(
    page_title="TBRGS – Traffic-based Route Guidance System",
    layout="wide",
)

st.title("TBRGS – Traffic-based Route Guidance System")
st.markdown(
    """
Interactive demo for your traffic-based route guidance system.

- **Model**: LSTM / GRU / CNN traffic flow predictor  
- **Routing algorithms**: GBFS, DFS, UCS, A*  
- **Data**: SCATS detectors mapped to road network nodes
"""
)

# Make sure base map exists
base_map_path = ensure_base_map()

# Load SCATS IDs and descriptions
scats_ids, desc_map = load_scats_and_descriptions()
weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# -----------------------------
# Init defaults in session_state
# -----------------------------
if "origin_id" not in st.session_state and scats_ids:
    st.session_state.origin_id = scats_ids[0]
if "dest_id" not in st.session_state and scats_ids:
    st.session_state.dest_id = scats_ids[1] if len(scats_ids) > 1 else scats_ids[0]
if "hour" not in st.session_state:
    st.session_state.hour = 9
if "minute" not in st.session_state:
    st.session_state.minute = "30"
if "weekday" not in st.session_state:
    st.session_state.weekday = 4  # Friday
if "model" not in st.session_state:
    st.session_state.model = "LSTM"
if "algorithm" not in st.session_state:
    st.session_state.algorithm = "astar"

# ------------- NEW: randomize callback -------------

def randomize_inputs():
    """Callback for the Randomize button – allowed to modify session_state."""
    if len(scats_ids) < 2:
        return
    o, d = random.sample(scats_ids, 2)
    st.session_state.origin_id = o
    st.session_state.dest_id = d
    st.session_state.hour = random.randint(0, 23)
    st.session_state.minute = random.choice(["00", "15", "30", "45"])
    st.session_state.weekday = random.randint(0, 6)
    # 可选：强制刷新一次
    # st.experimental_rerun()

# -----------------------------
# Layout: controls (left) & map (right)
# -----------------------------

left_col, right_col = st.columns([3, 7])

with left_col:
    st.subheader("Input Parameters")

    model = st.selectbox(
        "Model",
        ["LSTM", "GRU", "CNN"],
        key="model",
    )

    algorithm = st.selectbox(
        "Search Algorithm",
        ["gbfs", "ucs", "astar"],
        key="algorithm",
    )

    def format_scats(sid: int) -> str:
        return f"{sid} – {desc_map.get(sid, 'Unknown')}"

    origin_id = st.selectbox(
        "Origin SCATS",
        scats_ids,
        format_func=format_scats,
        key="origin_id",
    )

    dest_id = st.selectbox(
        "Destination SCATS",
        scats_ids,
        format_func=format_scats,
        key="dest_id",
    )

    st.markdown("**Time (HH:MM)**")
    hour = st.number_input(
        "Hour",
        min_value=0,
        max_value=23,
        step=1,
        key="hour",
    )
    minute = st.radio(
        "Minute",
        options=["00", "15", "30", "45"],
        key="minute",
        horizontal=True,
    )

    weekday_idx = st.radio(
        "Weekday",
        options=list(range(7)),
        format_func=lambda i: weekday_labels[i],
        key="weekday",
        horizontal=True,
    )

    st.markdown("---")
    # 关键改动：用 on_click 调用 randomize_inputs，而不是 if randomize: 再赋值
    st.button("🎲 Randomize Inputs", on_click=randomize_inputs)
    run_clicked = st.button("🚀 Run Prediction")

    results = None
    map_path = base_map_path

    if run_clicked:
        if origin_id == dest_id:
            st.error("Origin and destination cannot be the same.")
        else:
            time_str = f"{int(hour):02d}:{minute}"
            st.info(f"Running prediction for {weekday_labels[weekday_idx]} at {time_str} …")
            try:
                results = run_prediction_logic(
                    model=model,
                    weekday=weekday_idx,
                    time_str=time_str,
                    origin=int(origin_id),
                    destination=int(dest_id),
                    algorithm=algorithm,
                    top_k=5,
                )

                if not results:
                    st.warning("❌ No path found for the given parameters.")
                else:
                    st.subheader("Route Results")
                    for i, route in enumerate(results, start=1):
                        st.write(
                            f"**Route {i}**: "
                            f"{' → '.join(map(str, route['path']))} "
                            f" | Cost: `{route['cost']}` min"
                        )

                    all_paths = [r["path"] for r in results]
                    all_costs = [r["cost"] for r in results]

                    map_path = "data/output_map.html"
                    generate_map(
                        path_nodes_list=all_paths,
                        path_cost_list=all_costs,
                        pbf_file="data/B.osm.pbf",
                        path_file="data/Paths&Node_mapped.csv",
                        node_file="data/mapped_scats_nodes.csv",
                        output_html=map_path,
                    )
            except Exception as e:
                st.error(f"Prediction failed: {e}")

with right_col:
    st.subheader("Map View")
    try:
        html = load_map_html(map_path)
        components.html(html, height=650, scrolling=True)
    except FileNotFoundError:
        st.warning("Map file not found. Run a prediction first or check data/output_map.html.")
