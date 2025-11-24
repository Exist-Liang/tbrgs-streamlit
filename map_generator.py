import re
import itertools
from functools import lru_cache

import pandas as pd
import folium
from pyrosm import OSM


@lru_cache(maxsize=1)
def _get_osm_coords(pbf_file: str):
    """
    解析 OSM PBF，返回 {node_id: {"lat": ..., "lon": ...}} 字典。
    只在第一次调用时解析，之后都复用缓存。
    """
    osm = OSM(pbf_file)
    nodes_df, _ = osm.get_network(nodes=True, network_type="driving")
    return nodes_df.set_index("id")[["lat", "lon"]].to_dict("index")


@lru_cache(maxsize=1)
def _get_scats_coord(node_file: str):
    """
    载入 SCATS 节点表，返回 {Node_ID: {"Lat": ..., "Lon": ...}}。
    """
    scats_nodes = pd.read_csv(node_file)
    scats_nodes = scats_nodes.rename(
        columns={
            "Mapped_Node_ID": "Node_ID",
            "Mapped_Lon": "Lon",
            "Mapped_Lat": "Lat",
        }
    )
    return scats_nodes.set_index("Node_ID")[["Lat", "Lon"]].to_dict("index")


@lru_cache(maxsize=1)
def _get_path_df(path_file: str):
    """载入 Paths&Node_mapped.csv。"""
    return pd.read_csv(path_file)


def _compute_center_from_scats(scats_coord: dict):
    if not scats_coord:
        # 兜底：Boroondara 大致中心
        return -37.82, 145.05
    lats = [v["Lat"] for v in scats_coord.values()]
    lons = [v["Lon"] for v in scats_coord.values()]
    return sum(lats) / len(lats), sum(lons) / len(lons)


def generate_base_map(
    output_html: str = "data/output_map.html",
    pbf_file: str = "data/B.osm.pbf",
    node_file: str = "data/mapped_scats_nodes.csv",
):
    """
    只画出所有 SCATS 节点的底图，用于初始展示。
    """
    print("🧱 Initializing base map (nodes only)")

    scats_coord = _get_scats_coord(node_file)
    center_lat, center_lon = _compute_center_from_scats(scats_coord)

    fmap = folium.Map(location=(center_lat, center_lon), zoom_start=13, tiles="cartodbpositron")

    # 画出所有 SCATS 节点
    for node_id, coord in scats_coord.items():
        folium.CircleMarker(
            location=(coord["Lat"], coord["Lon"]),
            radius=3,
            color="gray",
            fill=True,
            fill_opacity=0.6,
            tooltip=f"SCATS Node {node_id}",
        ).add_to(fmap)

    fmap.save(output_html)
    print(f"✅ Base map generated: {output_html}")


def generate_map(
    path_nodes_list=None,
    path_cost_list=None,
    pbf_file: str = "data/B.osm.pbf",
    path_file: str = "data/Paths&Node_mapped.csv",
    node_file: str = "data/mapped_scats_nodes.csv",
    output_html: str = "data/output_map.html",
):
    """
    根据给定的路径节点序列，在底图上叠加高亮路线。
    path_nodes_list: 形如 [[n1, n2, n3, ...], [...]] 的列表。
    path_cost_list: 形如 [cost1, cost2, ...] 的列表，对应每条路径的时间。
    """
    print("🗺️ Generating map with routes...")

    scats_coord = _get_scats_coord(node_file)
    osm_coords = _get_osm_coords(pbf_file)
    path_df = _get_path_df(path_file)

    center_lat, center_lon = _compute_center_from_scats(scats_coord)
    fmap = folium.Map(
    location=(center_lat, center_lon),
    zoom_start=13,
    tiles="OpenStreetMap",
)

    # 先画出所有 SCATS 节点（灰色小点）
    for node_id, coord in scats_coord.items():
        folium.CircleMarker(
            location=(coord["Lat"], coord["Lon"]),
            radius=3,
            color="lightgray",
            fill=True,
            fill_opacity=0.6,
        ).add_to(fmap)

    # 将每条路径展开为若干 OSM polyline 段
    all_edge_groups = []

    if path_nodes_list:
        for path_nodes in path_nodes_list:
            # 清理 NaN，并转成 int
            cleaned = [int(n) for n in path_nodes if pd.notna(n)]
            segments = []

            for src, tgt in zip(cleaned, cleaned[1:]):
                matched = path_df[(path_df["Source"] == src) & (path_df["Target"] == tgt)]
                if matched.empty:
                    continue

                raw_str = str(matched.iloc[0].get("Path Nodes", ""))
                osm_ids = [int(m) for m in re.findall(r"\d+", raw_str)]
                coords = [
                    (osm_coords[n]["lat"], osm_coords[n]["lon"])
                    for n in osm_ids
                    if n in osm_coords
                ]
                if len(coords) >= 2:
                    segments.append(coords)

            all_edge_groups.append(segments)

    # 画出彩色路径
    print("🛣️ Drawing actual paths...")
    color_cycle = itertools.cycle(
        ["blue", "green", "purple", "orange", "darkred", "cadetblue", "black"]
    )

    for path_index, segments in enumerate(all_edge_groups):
        color = next(color_cycle)
        tooltip_text = f"Route {path_index + 1}"
        if path_cost_list and path_index < len(path_cost_list):
            tooltip_text += f" | Time: {round(path_cost_list[path_index], 2)} min"

        for seg in segments:
            folium.PolyLine(
                seg,
                color=color,
                weight=8,
                opacity=0.85,
                tooltip=tooltip_text,
            ).add_to(fmap)

    # 高亮路径上的节点（起点/终点/中间）
    print("🌟 Highlighting path nodes...")
    if path_nodes_list:
        for path_nodes in path_nodes_list:
            cleaned = [int(n) for n in path_nodes if pd.notna(n)]
            for i, n in enumerate(cleaned):
                if n not in scats_coord:
                    continue
                coord = scats_coord[n]
                if i == 0:
                    label = "Start"
                    color = "green"
                elif i == len(cleaned) - 1:
                    label = "End"
                    color = "red"
                else:
                    label = f"Node {n}"
                    color = "blue"

                folium.Marker(
                    location=(coord["Lat"], coord["Lon"]),
                    tooltip=label,
                    icon=folium.Icon(color=color),
                ).add_to(fmap)

    fmap.save(output_html)
    print(f"✅ Map generated: {output_html}")
