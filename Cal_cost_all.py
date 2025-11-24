import pandas as pd
import numpy as np
import math
import argparse
from datetime import datetime
from functools import lru_cache

import torch
import torch.nn as nn


# ==== 模型定义 ====

class TrafficLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        # 只取最后一个时间步
        return self.fc(out[:, -1, :]).squeeze(-1)


class TrafficGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class TrafficCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入 [batch, seq_len, 4] → 在 forward 里转成 [batch, 4, seq_len]
        self.conv = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, 4] → [batch, 4, seq_len]
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze(-1)


# ==== 工具函数 ====

def flow_to_speed(flow: float) -> float:
    """由流量估计车速（km/h），基于你原来的经验公式。"""
    if flow <= 0:
        return 60.0
    try:
        a, b, c = -1.4648375, 93.75, -flow
        delta = b ** 2 - 4 * a * c
        if delta < 0:
            return 10.0
        r1 = (-b + math.sqrt(delta)) / (2 * a)
        r2 = (-b - math.sqrt(delta)) / (2 * a)
        # 保留原有分段逻辑
        if flow <= 351 and r2 > 0:
            return min(r2, 60.0)
        if 0 < r1 <= 60:
            return r1
        return 10.0
    except Exception:
        return 10.0


def calculate_travel_time_km(distance_km: float, flow: float) -> float:
    """根据距离(km)和流量估计行程时间（分钟）。"""
    speed = flow_to_speed(flow)  # km/h
    if speed <= 0:
        return 60.0
    # 小时 → 分钟
    return (distance_km / speed) * 60.0


# ==== 缓存：模型与历史数据 ====

@lru_cache(maxsize=3)
def _get_model(model_type: str, model_path: str):
    """带缓存地创建并加载一次模型，后续重复使用。"""
    mt = model_type.upper()
    if mt == "LSTM":
        model = TrafficLSTM(4, 64, 2, 0.2)
    elif mt == "GRU":
        model = TrafficGRU(4, 64, 2, 0.2)
    elif mt == "CNN":
        model = TrafficCNN()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


@lru_cache(maxsize=1)
def _get_mapping_df(mapping_path: str = "data/location_id_mapping.csv") -> pd.DataFrame:
    return pd.read_csv(mapping_path)


@lru_cache(maxsize=1)
def _get_history_df() -> pd.DataFrame:
    """读取并预处理带 Location Code 的历史流量数据。"""
    df = pd.read_csv("data/cleaned_scats_data_with_location_code.csv", parse_dates=["DateTime"])
    df["Weekday"] = df["DateTime"].dt.weekday
    df["Hour"] = df["DateTime"].dt.hour + df["DateTime"].dt.minute / 60.0
    return df


@lru_cache(maxsize=1)
def _get_history_with_ids() -> pd.DataFrame:
    """历史数据 + Location ID，一次性 merge，供预测使用。"""
    history_df = _get_history_df()
    mapping_df = _get_mapping_df()
    # merge 后会有 Location ID 列
    return history_df.merge(mapping_df, on="Location Code", how="left")


# ==== 预测函数 ====

def predict_flow_real_time(
    model_type,
    model_path,
    location_ids,
    weekday,
    hour,
    mapping_df=None,
):
    """
    对给定的 Edge / Location Code 列表做实时流量预测。
    返回 {location_code: predicted_volume}.
    """
    model = _get_model(model_type, model_path)
    history_df = _get_history_with_ids()
    if mapping_df is None:
        mapping_df = _get_mapping_df()

    predictions = {}

    for loc_code in location_ids:
        # 兼容可能的类型差异
        try:
            mask = mapping_df["Location Code"] == loc_code
        except Exception:
            mask = mapping_df["Location Code"].astype(str) == str(loc_code)

        row = mapping_df[mask]
        if row.empty:
            # 映射不到 ID，就给一个很小的流量
            predictions[loc_code] = 1e-3
            continue

        loc_numeric = row["Location ID"].values[0]

        loc_hist = history_df[history_df["Location ID"] == loc_numeric]
        if loc_hist.empty:
            predictions[loc_code] = 1e-3
            continue

        # 优先使用同 weekday 且在 hour±1 小时内的数据
        past_data = loc_hist[
            (loc_hist["Weekday"] == weekday)
            & (loc_hist["Hour"].between(hour - 1.0, hour + 1.0))
        ]

        if len(past_data) < 16:
            # 不足则退化为同 weekday
            past_data = loc_hist[loc_hist["Weekday"] == weekday]

        if len(past_data) < 16:
            # 还是不足就用全部历史
            past_data = loc_hist

        if past_data.empty:
            predictions[loc_code] = 1e-3
            continue

        # 构造 16 条序列样本
        past_data = past_data.sort_values("DateTime")
        if len(past_data) >= 16:
            sample = past_data.tail(16)
        else:
            sample = past_data.sample(n=16, replace=True).sort_values("DateTime")

        input_np = np.stack(
            [
                sample["Volume"].values,
                sample["Weekday"].values,
                sample["Hour"].values,
                sample["Location ID"].values,
            ],
            axis=1,
        ).reshape(1, 16, 4)

        input_tensor = torch.tensor(input_np, dtype=torch.float32)

        with torch.no_grad():
            raw_pred = float(model(input_tensor).item())

        if raw_pred > 0.005:
            predictions[loc_code] = raw_pred
        else:
            # 回退到同 weekday + 接近 hour 的平均
            ref = loc_hist[
                (loc_hist["Weekday"] == weekday)
                & (np.isclose(loc_hist["Hour"], hour))
            ]
            if not ref.empty and not np.isnan(ref["Volume"].mean()):
                predictions[loc_code] = float(ref["Volume"].mean())
            else:
                predictions[loc_code] = 1e-3

    return predictions


# ==== 主流程：给边表加上 Travel_Time_Min ====

def assign_travel_times_return(
    edges_path: str,
    mapping_path: str,
    model_type: str,
    model_path: str,
    weekday: int,
    hour: float,
    location_ids,
) -> pd.DataFrame:
    """
    读取边表 + 路径长度 + 模型预测结果，返回带 Travel_Time_Min 的 DataFrame。
    """
    edges = pd.read_csv(edges_path)
    mapping_df = pd.read_csv(mapping_path)

    # 路径长度表，单位：米
    path_info = pd.read_csv("data/Paths&Node_mapped.csv")
    path_length_map = dict(zip(path_info["Edge ID"], path_info["Path Length (m)"]))

    # 流量预测（location_ids 通常是 Edge ID / Location Code 列表）
    flow_map = predict_flow_real_time(
        model_type=model_type,
        model_path=model_path,
        location_ids=location_ids,
        weekday=weekday,
        hour=hour,
        mapping_df=mapping_df,
    )

    distance_m = []
    travel_times = []

    for _, row in edges.iterrows():
        edge_id = row["Edge ID"]
        dist_m = path_length_map.get(edge_id, np.nan)
        flow = flow_map.get(edge_id, 0.0)

        if not np.isnan(dist_m):
            tt = calculate_travel_time_km(dist_m / 1000.0, flow)
        else:
            tt = np.nan

        distance_m.append(dist_m)
        travel_times.append(tt)

    edges["Distance_m"] = distance_m
    edges["Predicted Flow"] = edges["Edge ID"].map(flow_map)
    edges["Travel_Time_Min"] = travel_times

    return edges


# ==== 命令行入口（可选） ====

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🧠 Real-time traffic model predictor")
    parser.add_argument("--model", type=str, choices=["LSTM", "GRU", "CNN"], required=True)
    parser.add_argument("--time", type=str, required=True, help="Time string like '09:30'")
    parser.add_argument("--weekday", type=int, choices=range(7), required=True)
    parser.add_argument(
        "--location_ids",
        type=str,
        nargs="+",
        required=True,
        help="Location Code / Edge ID list to predict",
    )

    args = parser.parse_args()
    dt = datetime.strptime(args.time, "%H:%M")

    df_with_times = assign_travel_times_return(
        edges_path="data/Paths&Node_mapped.csv",
        mapping_path="data/location_id_mapping.csv",
        model_type=args.model,
        model_path=f"ML/best_{args.model.lower()}_model.pth",
        weekday=args.weekday,
        hour=dt.hour + dt.minute / 60.0,
        location_ids=args.location_ids,
    )

    print(df_with_times[df_with_times["Edge ID"].isin(args.location_ids)])
