import heapq

import pandas as pd

from Cal_cost_all import assign_travel_times_return
from algorithm import gbfs, bfs, astar_k_paths


# === 提供给 GUI / Streamlit 调用的主函数 ===

def run_prediction_logic(model, weekday, time_str, origin, destination, algorithm="astar", top_k=1):
    """
    高层封装：
    - 根据给定的模型和时间，先为所有边预测 Travel_Time_Min
    - 然后在预测后的图上，用指定算法搜索从 origin 到 destination 的路径
    """
    EDGES_CSV = "data/Paths&Node_mapped.csv"
    MAPPING_CSV = "data/location_id_mapping.csv"
    MODEL_PATH = f"ML/best_{model.lower()}_model.pth"

    dt = pd.to_datetime(time_str, format="%H:%M")
    hour = dt.hour + dt.minute / 60.0

    mapping_df = pd.read_csv(MAPPING_CSV)
    location_ids = mapping_df["Location Code"].tolist()

    # 1) 预测边的行程时间
    predicted_df = assign_travel_times_return(
        edges_path=EDGES_CSV,
        mapping_path=MAPPING_CSV,
        model_type=model,
        model_path=MODEL_PATH,
        weekday=weekday,
        hour=hour,
        location_ids=location_ids,
    )

    # 2) 构建图
    graph = build_graph_from_df(predicted_df)
    origin_node = scats_to_graph_node(origin, predicted_df)
    destination_node = scats_to_graph_node(destination, predicted_df)

    # 3) 根据选择的算法搜索路径
    if algorithm == "gbfs":
        goal_node, count, path = gbfs(graph, origin_node, [destination_node])
        routes = [(evaluate_path(graph, path), path)] if path else []
    elif algorithm == "bfs":
        goal_node, count, path = bfs(graph, origin_node, [destination_node])
        routes = [(evaluate_path(graph, path), path)] if path else []
    elif algorithm == "ucs":
        # 简单的多路径 UCS，内部已经是按 cost 升序扩展
        routes = find_multiple_routes(graph, origin_node, [destination_node], top_k=top_k)
        # 加上每条边 0.5 分钟的 penalty，使得路径更短更优
        routes = [(cost + 0.5 * (len(path) - 1), path) for cost, path in routes]
    elif algorithm == "astar":
        # 这里假设 astar_k_paths 返回 [{ "path": [...], "cost": ... }, ...]
        raw_routes = astar_k_paths(graph, origin_node, [destination_node], k=top_k)
        routes = [(evaluate_path(graph, r["path"]), r["path"]) for r in raw_routes]
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # 4) 格式化输出
    return [
        {"path": path, "cost": round(total_time, 2)}
        for total_time, path in routes
    ]


# === 图构建 & 工具函数 ===

def build_graph_from_df(df):
    """
    根据带 Travel_Time_Min 的 DataFrame 构建有向图：
    graph[src] = [(tgt, time), ...]
    """
    graph = {}
    for _, row in df.iterrows():
        if pd.notna(row.get("Travel_Time_Min")):
            src = row["Source"]
            tgt = row["Target"]
            time = row["Travel_Time_Min"]
            graph.setdefault(src, []).append((tgt, time))
    return graph


def scats_to_graph_node(scats_number, df_edges: pd.DataFrame):
    """
    根据 SCATS 编号，查找对应的图节点 ID（Source/Target）。
    """
    source_match = df_edges[df_edges["SCATS_Source"] == scats_number]["Source"].unique()
    if len(source_match) > 0:
        return source_match[0]

    target_match = df_edges[df_edges["SCATS_Target"] == scats_number]["Target"].unique()
    if len(target_match) > 0:
        return target_match[0]

    raise ValueError(f"SCATS number {scats_number} not found in edge data.")


def evaluate_path(graph, path):
    """
    根据图和路径，累加边的 Travel_Time_Min，并加上 0.5 * (边数) 的 penalty。
    """
    if not path or len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        src = path[i]
        tgt = path[i + 1]
        edge_list = graph.get(src, [])
        for n, t in edge_list:
            if n == tgt:
                total += t
                break
    # penalty：每条边额外 0.5 分钟
    total += 0.5 * (len(path) - 1)
    return total


def find_multiple_routes(graph, origin, destinations, top_k=3):
    """
    简单的多路径 Uniform Cost Search（类似 K 最短路径的近似版）。
    """
    heap = []
    # 元组：(cost, sequence, current_node, path)
    heapq.heappush(heap, (0.0, 0, origin, [origin]))
    visited_paths = []
    sequence = 1

    target_set = set(destinations)

    while heap and len(visited_paths) < top_k:
        cost, _, current, path = heapq.heappop(heap)

        if current in target_set:
            visited_paths.append((cost, path))
            continue

        for neighbor, edge_cost in graph.get(current, []):
            if neighbor not in path:
                heapq.heappush(
                    heap,
                    (cost + edge_cost, sequence, neighbor, path + [neighbor]),
                )
                sequence += 1

    return visited_paths


# === CLI 使用（可选） ===

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Route prediction CLI")
    parser.add_argument("--model", type=str, choices=["LSTM", "GRU", "CNN"], default="LSTM")
    parser.add_argument("--weekday", type=int, required=True)
    parser.add_argument("--time", type=str, required=True, help="HH:MM")
    parser.add_argument("--origin", type=int, required=True, help="Origin SCATS number")
    parser.add_argument("--destination", type=int, required=True, help="Destination SCATS number")
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["gbfs", "bfs", "ucs", "astar"],
        default="astar",
    )
    parser.add_argument("--top_k", type=int, default=1)

    args = parser.parse_args()

    result = run_prediction_logic(
        model=args.model,
        weekday=args.weekday,
        time_str=args.time,
        origin=args.origin,
        destination=args.destination,
        algorithm=args.algorithm,
        top_k=args.top_k,
    )

    if result:
        print("\n=== Top Route(s) Found ===")
        for item in result:
            print(f"PATH_RESULT: {' -> '.join(map(str, item['path']))} | TIME={item['cost']:.2f} min")
    else:
        print("❌ No path found from origin to destination.")
