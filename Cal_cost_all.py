import pandas as pd
import numpy as np
import math
import argparse
from datetime import datetime
import torch
import torch.nn as nn

# ==== 模型定义 ====
class TrafficLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(TrafficLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()

class TrafficGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(TrafficGRU, self).__init__()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze()

class TrafficCNN(nn.Module):
    def __init__(self):
        super(TrafficCNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, channels, seq_len]
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze()
# ==== 工具函数 ====

def flow_to_speed(flow):
    if flow <= 0: return 60
    try:
        a, b, c = -1.4648375, 93.75, -flow
        delta = b**2 - 4*a*c
        if delta < 0: return 10
        r1 = (-b + math.sqrt(delta)) / (2*a)
        r2 = (-b - math.sqrt(delta)) / (2*a)
        return min(r2, 60) if flow <= 351 and r2 > 0 else r1 if 0 < r1 <= 60 else 10
    except: return 10

def calculate_travel_time_km(distance_km, flow):
    speed = flow_to_speed(flow)
    return (distance_km / speed if speed > 0 else 1) * 60

# ==== 预测函数 ====
def predict_flow_real_time(model_type, model_path, location_ids, weekday, hour, mapping_df=None):
    model_type = model_type.upper()
    if model_type == 'LSTM':
        model = TrafficLSTM(4, 64, 2, 0.2)
    elif model_type == 'GRU':
        model = TrafficGRU(4, 64, 2, 0.2)
    elif model_type == 'CNN':
        model = TrafficCNN()
    else:
        raise ValueError("Unsupported model type")

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    history_df = pd.read_csv("data/cleaned_scats_data_with_location_code.csv", parse_dates=["DateTime"])
    if mapping_df is None:
        mapping_df = pd.read_csv("data/location_id_mapping.csv")
    history_df = history_df.merge(mapping_df, on="Location Code", how="left")
    history_df["Weekday"] = history_df["DateTime"].dt.weekday
    history_df["Hour"] = history_df["DateTime"].dt.hour + history_df["DateTime"].dt.minute / 60

    predictions = {}
    with torch.no_grad():
        for loc_id in location_ids:
            if loc_id not in mapping_df["Location Code"].values:
                print(f"❌ {loc_id} not found in mapping.")
                continue

            loc_numeric = mapping_df[mapping_df["Location Code"] == loc_id]["Location ID"].values[0]
            past_data = history_df[
                (history_df["Location ID"] == loc_numeric) &
                (history_df["Weekday"] == weekday) &
                (history_df["Hour"].between(hour - 1, hour + 1))
            ]

            if len(past_data) < 16:
                past_data = history_df[
                    (history_df["Location ID"] == loc_numeric) &
                    (history_df["Weekday"] == weekday)
                ]

            sample = past_data.sort_values("DateTime").sample(n=16, replace=True) if not past_data.empty else pd.DataFrame()

            if sample.empty:
                ref = history_df[(history_df["Location ID"] == loc_numeric) & 
                                 (history_df["Weekday"] == weekday) &
                                 (np.isclose(history_df["Hour"], hour))]
                predictions[loc_id] = ref["Volume"].mean() if not ref.empty else 1e-3
                continue

            input_np = np.stack([
                sample["Volume"].values,
                sample["Weekday"].values,
                sample["Hour"].values,
                sample["Location ID"].values
            ], axis=1).reshape(1, 16, 4)

            if model_type == "CNN":
                input_np = input_np.reshape(1, 16, 4)

            input_tensor = torch.tensor(input_np, dtype=torch.float32)
            raw_pred = model(input_tensor).item()
            predictions[loc_id] = raw_pred if raw_pred > 0.005 else (
                history_df[(history_df["Location ID"] == loc_numeric) &
                           (history_df["Weekday"] == weekday) &
                           (np.isclose(history_df["Hour"], hour))]["Volume"].mean() or 1e-3
            )

    return predictions

# ==== 主流程 ====
def assign_travel_times_return(edges_path, mapping_path, model_type, model_path, weekday, hour, location_ids):
    edges = pd.read_csv(edges_path)
    mapping_df = pd.read_csv(mapping_path)
    
    # ✅ 加载路径长度表（你上传的 Paths&Node_mapped.csv）
    path_info = pd.read_csv("data/Paths&Node_mapped.csv")  # 确保与实际路径一致
    path_length_map = dict(zip(path_info["Edge ID"], path_info["Path Length (m)"]))

    # 获取流量预测
    flow_map = predict_flow_real_time(model_type, model_path, location_ids, weekday, hour, mapping_df)

    # 生成新的距离和时间字段
    distance_m, travel_times = [], []
    for _, row in edges.iterrows():
        loc = row["Edge ID"]  # Edge ID（形如 A001）
        dist = path_length_map.get(loc, np.nan)  # 单位：米
        flow = flow_map.get(loc, 0)
        travel_time = calculate_travel_time_km(dist / 1000, flow) if not np.isnan(dist) else np.nan  # 转 km 后代入计算

        distance_m.append(dist)
        travel_times.append(travel_time)

    edges["Distance_m"] = distance_m
    edges["Predicted Flow"] = edges["Edge ID"].map(flow_map)
    edges["Travel_Time_Min"] = travel_times
    return edges

# ==== 命令行入口 ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🧠 Real-time traffic model predictor")
    parser.add_argument('--model', type=str, choices=['LSTM', 'GRU', 'CNN'], required=True)
    parser.add_argument('--time', type=str, required=True, help="Time string like '09:30'")
    parser.add_argument('--weekday', type=int, choices=range(7), required=True)
    parser.add_argument('--location_ids', type=str, nargs='+', required=True)

    args = parser.parse_args()
    dt = datetime.strptime(args.time, "%H:%M")

    df_with_times = assign_travel_times_return(
        edges_path="data/Paths&Node_mapped.csv",
        mapping_path="data/location_id_mapping.csv",
        model_type=args.model,
        model_path=f"ML/best_{args.model.lower()}_model.pth",
        weekday=args.weekday,
        hour=dt.hour + dt.minute / 60,
        location_ids=args.location_ids
    )

    print(df_with_times[df_with_times["Edge ID"].isin(args.location_ids)])
