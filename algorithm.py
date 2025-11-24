# === algorithm.py ===
# Contains Part A routing algorithms (BFS, DFS, GBFS, UCS, A*)

import heapq
from collections import deque
import math

# === Breadth-First Search (BFS) ===
def bfs(edges, origin, destinations):
    queue = deque()
    queue.append((origin, [origin]))
    visited = set([origin])
    count = 1

    while queue:
        current, path = queue.popleft()
        if current in destinations:
            return current, count, path

        for neighbor, _ in sorted(edges.get(current, []), key=lambda x: x[0]):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
                count += 1

    return None, count, []

# === Depth-First Search (DFS) ===
def dfs(edges, origin, destinations):
    stack = [(origin, [origin], 0)]
    count = 1
    min_cost = float('inf')
    best_path = None
    goal_node = None

    while stack:
        current, path, cost = stack.pop()
        if current in destinations:
            if cost < min_cost:
                min_cost = cost
                best_path = path
                goal_node = current
            continue

        for neighbor, edge_cost in reversed(sorted(edges.get(current, []), key=lambda x: x[0])):
            if neighbor not in path:
                stack.append((neighbor, path + [neighbor], cost + edge_cost))
                count += 1

    return goal_node, count, best_path

# === Uniform Cost Search (UCS / cus1) ===
def cus1(edges, origin, destinations):
    heap = []
    heapq.heappush(heap, (0, 0, origin, [origin]))
    visited = {}
    count = 0
    sequence = 1
    best_path = None
    goal_node = None

    while heap:
        cost, _, current, path = heapq.heappop(heap)
        count += 1

        if current in destinations:
            if goal_node is None or cost < visited.get(goal_node, float('inf')):
                goal_node = current
                best_path = path
            continue

        for neighbor, edge_cost in sorted(edges.get(current, []), key=lambda x: (x[0], x[1])):
            new_cost = cost + edge_cost
            if neighbor not in visited or new_cost < visited[neighbor]:
                visited[neighbor] = new_cost
                heapq.heappush(heap, (new_cost, sequence, neighbor, path + [neighbor]))
                sequence += 1

    return goal_node, count, best_path

# === Greedy Best-First Search (GBFS) ===
def gbfs(nodes_edges, origin, destinations):
    def heuristic(node, goal):
        return 0  # Replace with real heuristic if needed

    visited = set()
    count = 0
    sequence = 0
    pq = []

    for dest in destinations:
        heapq.heappush(pq, (heuristic(origin, dest), sequence, origin, [origin]))
        sequence += 1

    while pq:
        h, _, current, path = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)
        count += 1

        if current in destinations:
            return current, count, path

        for neighbor, time in sorted(nodes_edges.get(current, []), key=lambda x: x[0]):
            if neighbor not in visited:
                heapq.heappush(pq, (heuristic(neighbor, destinations[0]), sequence, neighbor, path + [neighbor]))
                sequence += 1

    return None, count, []

# === A* Search (A-Star) ===
def astar_k_paths(graph, origin, destinations, k=5):
    def heuristic(n1, n2):
        return 0  # 你可以根据实际经纬度坐标设置启发函数

    open_list = []
    heapq.heappush(open_list, (0, 0, origin, [origin], 0))  # (f, seq, node, path, g)
    sequence = 1
    found_paths = []
    visited_paths = set()

    while open_list and len(found_paths) < k:
        f, _, current, path, g = heapq.heappop(open_list)

        if current in destinations:
            path_tuple = tuple(path)
            if path_tuple not in visited_paths:
                visited_paths.add(path_tuple)
                found_paths.append({
                    "path": path,
                    "cost": round(g, 2)
                })
            continue

        for neighbor, cost in graph.get(current, []):
            if neighbor in path:
                continue  # 避免环
            new_g = g + cost
            new_f = new_g + heuristic(neighbor, destinations[0])
            heapq.heappush(open_list, (new_f, sequence, neighbor, path + [neighbor], new_g))
            sequence += 1

    return found_paths