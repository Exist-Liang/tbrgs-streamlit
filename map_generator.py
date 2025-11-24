import pandas as pd
import folium
from pyrosm import OSM
import re
import itertools

def generate_base_map(output_html="data/output_map.html",
                      pbf_file="data/B.osm.pbf",
                      node_file="data/mapped_scats_nodes.csv"):
    print("🧱 Initializing base map (nodes only)")
    osm = OSM(pbf_file)
    nodes_df, _ = osm.get_network(nodes=True, network_type="driving")

    scats_nodes = pd.read_csv(node_file)
    scats_nodes.rename(columns={
        "Mapped_Node_ID": "Node_ID",
        "Mapped_Lon": "Lon",
        "Mapped_Lat": "Lat"
    }, inplace=True)
    scats_coord = scats_nodes.set_index("Node_ID")[["Lat", "Lon"]].to_dict("index")

    center = list(scats_coord.values())[0]
    fmap = folium.Map(location=[center["Lat"], center["Lon"]], zoom_start=14)

    for node_id, coord in scats_coord.items():
        folium.CircleMarker(
            location=(coord["Lat"], coord["Lon"]),
            radius=4,
            color="red",
            fill=True,
            tooltip=f"Node {node_id}"
        ).add_to(fmap)

    fmap.save(output_html)
    print(f"✅ Base map generated: {output_html}")

def generate_map(path_nodes_list=None,
                 path_cost_list=None,
                 pbf_file="data/B.osm.pbf",
                 path_file="data/Paths&Node_mapped.csv",
                 node_file="data/mapped_scats_nodes.csv",
                 output_html="data/output_map.html"):
    print("📆 Loading OSM road network...")
    osm = OSM(pbf_file)
    nodes_df, _ = osm.get_network(nodes=True, network_type="driving")
    osm_coords = nodes_df.set_index("id")[["lon", "lat"]].to_dict("index")

    print("📍 Loading SCATS node coordinates...")
    scats_nodes = pd.read_csv(node_file)
    scats_nodes.rename(columns={
        "Mapped_Node_ID": "Node_ID",
        "Mapped_Lon": "Lon",
        "Mapped_Lat": "Lat"
    }, inplace=True)
    scats_coord = scats_nodes.set_index("Node_ID")[["Lat", "Lon"]].to_dict("index")

    print("🔗 Loading CSV path data...")
    path_df = pd.read_csv(path_file)
    all_edge_groups = []

    if path_nodes_list:
        for path_index, path_nodes in enumerate(path_nodes_list):
            node_pairs = [(int(path_nodes[i]), int(path_nodes[i+1])) for i in range(len(path_nodes)-1) if pd.notna(path_nodes[i]) and pd.notna(path_nodes[i+1])]
            path_segments = []
            for src, tgt in node_pairs:
                matched_row = path_df[(path_df["Source"] == src) & (path_df["Target"] == tgt)]
                if not matched_row.empty:
                    raw_str = matched_row.iloc[0].get("Path Nodes", "")
                    row_ids = [int(n) for n in re.findall(r"\d+", raw_str)]
                    coords = [(osm_coords[n]["lat"], osm_coords[n]["lon"]) for n in row_ids if n in osm_coords]
                    if len(coords) >= 2:
                        path_segments.append(coords)
            all_edge_groups.append(path_segments)

    center = list(scats_coord.values())[0]
    fmap = folium.Map(location=[center["Lat"], center["Lon"]], zoom_start=14)

    print("📍 Drawing SCATS nodes...")
    for node_id, coord in scats_coord.items():
        folium.CircleMarker(
            location=(coord["Lat"], coord["Lon"]),
            radius=4,
            color="red",
            fill=True,
            tooltip=f"Node {node_id}"
        ).add_to(fmap)

    print("🛣️ Drawing actual paths...")
    color_cycle = itertools.cycle([
        "blue", "green", "purple", "orange", "darkred", "cadetblue", "black"
    ])
    for path_index, group in enumerate(all_edge_groups):
        color = next(color_cycle)
        offset = (path_index - len(path_nodes_list) / 2) * 0.00005

        tooltip_text = f"Route {path_index + 1}"
        if path_cost_list and path_index < len(path_cost_list):
            tooltip_text += f" | Time: {round(path_cost_list[path_index], 2)} min"

        for seg in group:
            offset_seg = [(lat + offset, lon + offset) for lat, lon in seg]
            folium.PolyLine(offset_seg, color=color, weight=8, opacity=0.8, tooltip=tooltip_text).add_to(fmap)

    print("🌟 Highlighting path nodes...")
    if path_nodes_list:
        for path_nodes in path_nodes_list:
            for i, n in enumerate(path_nodes):
                if pd.isna(n):
                    continue
                n_int = int(n)
                if n_int in scats_coord:
                    coord = scats_coord[n_int]
                    folium.Marker(
                        location=(coord["Lat"], coord["Lon"]),
                        tooltip=("Start" if i == 0 else "End" if i == len(path_nodes) - 1 else f"Node {n}"),
                        icon=folium.Icon(color="green" if i == 0 else "red" if i == len(path_nodes) - 1 else "blue")
                    ).add_to(fmap)

    fmap.save(output_html)
    print(f"✅ Map generated: {output_html}")

