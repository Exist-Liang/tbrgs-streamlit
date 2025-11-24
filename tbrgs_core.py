import pandas as pd
from Cal_cost_all import assign_travel_times_return
from algorithm import gbfs, bfs, dfs, cus1, astar_k_paths
import heapq

# === 提供 GUI 端调用的主函数 ===
def run_prediction_logic(model, weekday, time_str, origin, destination, algorithm="astar", top_k=1):
    EDGES_CSV = "data/Paths&Node_mapped.csv"
    MAPPING_CSV = "data/location_id_mapping.csv"
    MODEL_PATH = f"ML/best_{model.lower()}_model.pth"

    dt = pd.to_datetime(time_str, format="%H:%M")
    hour = dt.hour + dt.minute / 60

    mapping_df = pd.read_csv(MAPPING_CSV)
    location_ids = mapping_df["Location Code"].tolist()

    predicted_df = assign_travel_times_return(
        edges_path=EDGES_CSV,
        mapping_path=MAPPING_CSV,
        model_type=model,
        model_path=MODEL_PATH,
        weekday=weekday,
        hour=hour,
        location_ids=location_ids
    )

    # 构建图结构
    graph = build_graph_from_df(predicted_df)
    origin_node = scats_to_graph_node(origin, predicted_df)
    destination_node = scats_to_graph_node(destination, predicted_df)

    # 搜索路径
    if algorithm == "gbfs":
        goal_node, count, path = gbfs(graph, origin_node, [destination_node])
        routes = [(evaluate_path(graph, path), path)] if path else []
    elif algorithm == "bfs":
        goal_node, count, path = bfs(graph, origin_node, [destination_node])
        routes = [(evaluate_path(graph, path), path)] if path else []
    elif algorithm == "dfs":
        goal_node, count, path = dfs(graph, origin_node, [destination_node])
        routes = [(evaluate_path(graph, path), path)] if path else []
    elif algorithm == "ucs":
        routes = find_multiple_routes(graph, origin_node, [destination_node], top_k=top_k)
        routes = [(cost + 0.5 * (len(path) - 1), path) for cost, path in routes]
    elif algorithm == "astar":
        routes = astar_k_paths(graph, origin_node, [destination_node], k=top_k)
        routes = [(evaluate_path(graph, r["path"]), r["path"]) for r in routes]
    else:
        raise ValueError("Unsupported algorithm")

    return [
        {"path": path, "cost": round(total_time, 2)}
        for total_time, path in routes
    ]

# === CLI 用途仍保留 ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["LSTM", "GRU", "CNN"], default="LSTM")
    parser.add_argument("--weekday", type=int, required=True)
    parser.add_argument("--time", type=str, required=True)
    parser.add_argument("--origin", type=int, required=True)
    parser.add_argument("--destination", type=int, required=True)
    parser.add_argument("--algorithm", type=str, choices=["gbfs", "bfs", "dfs", "ucs", "astar"], default="astar")
    parser.add_argument("--top_k", type=int, default=1)
    args = parser.parse_args()

    result = run_prediction_logic(
        model=args.model,
        weekday=args.weekday,
        time_str=args.time,
        origin=args.origin,
        destination=args.destination,
        algorithm=args.algorithm,
        top_k=args.top_k
    )

    if result:
        print("\n=== Top Route(s) Found ===")
        for item in result:
            print(f"PATH_RESULT: {' -> '.join(map(str, item['path']))} | TIME={item['cost']:.2f}")
    else:
        print("❌ No path found from origin to destination.")

# === 通用构件 ===
def build_graph_from_df(df):
    graph = {}
    for _, row in df.iterrows():
        if pd.notna(row["Travel_Time_Min"]):
            src = row["Source"]
            tgt = row["Target"]
            time = row["Travel_Time_Min"]
            graph.setdefault(src, []).append((tgt, time))
    return graph

def scats_to_graph_node(scats_number, df_edges):
    source_match = df_edges[df_edges["SCATS_Source"] == scats_number]["Source"].unique()
    if len(source_match) > 0:
        return source_match[0]
    target_match = df_edges[df_edges["SCATS_Target"] == scats_number]["Target"].unique()
    if len(target_match) > 0:
        return target_match[0]
    raise ValueError(f"SCATS number {scats_number} not found in edge data.")

def evaluate_path(graph, path):
    return sum(
        t for i in range(len(path) - 1)
        for n, t in graph[path[i]] if n == path[i + 1]
    ) + 0.5 * (len(path) - 1)

def find_multiple_routes(graph, origin, destinations, top_k=3):
    heap = []
    heapq.heappush(heap, (0, 0, origin, [origin]))
    visited_paths = []
    sequence = 1

    while heap and len(visited_paths) < top_k:
        cost, _, current, path = heapq.heappop(heap)

        if current in destinations:
            visited_paths.append((cost, path))
            continue

        for neighbor, edge_cost in graph.get(current, []):
            if neighbor not in path:
                heapq.heappush(heap, (cost + edge_cost, sequence, neighbor, path + [neighbor]))
                sequence += 1

    return visited_paths


