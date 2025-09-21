import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.utils.data import DataLoader, Dataset
import xgboost as xgb
import numpy as np

# -------------------------------
# 1. LSTM Model for Temporal Delivery Patterns
# -------------------------------
class DeliveryLSTM(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, num_layers=2, output_dim=32):
        super(DeliveryLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last hidden state
        return self.fc(out)  # [batch, output_dim]


# -------------------------------
# 2. GNN Model for Route Graphs
# -------------------------------
class RouteGNN(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=32, output_dim=32):
        super(RouteGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x  # node embeddings


# -------------------------------
# 3. ETA Predictor (combining LSTM + GNN features)
# -------------------------------
class ETAPredictor(nn.Module):
    def __init__(self, lstm_dim=32, gnn_dim=32, hidden_dim=64):
        super(ETAPredictor, self).__init__()
        self.fc1 = nn.Linear(lstm_dim + gnn_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # ETA in hours/days

    def forward(self, lstm_features, gnn_features):
        x = torch.cat([lstm_features, gnn_features], dim=-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# -------------------------------
# 4. Wrapper: Real-time ETA + Trust Prediction
# -------------------------------
class DeliveryPipeline:
    def __init__(self):
        self.lstm_model = DeliveryLSTM()
        self.gnn_model = RouteGNN()
        self.eta_model = ETAPredictor()
        self.trust_model = None  # trained separately (XGBoost)

    def predict_eta_distribution(self, seq_data, graph_x, edge_index, node_id):
        lstm_out = self.lstm_model(seq_data)
        gnn_out = self.gnn_model(graph_x, edge_index)
        route_feat = gnn_out[node_id].unsqueeze(0)  # embedding for specific route
        eta_pred = self.eta_model(lstm_out, route_feat)
        return eta_pred

    def train_trust_model(self, features, labels):
        dtrain = xgb.DMatrix(features, label=labels)
        params = {"objective": "binary:logistic", "eval_metric": "logloss"}
        self.trust_model = xgb.train(params, dtrain, num_boost_round=50)

    def predict_trust_score(self, features):
        dtest = xgb.DMatrix(features)
        return self.trust_model.predict(dtest)

    def update_with_realtime_data(self, eta_pred, traffic_factor=1.0, weather_factor=1.0):
        # adjust ETA based on real-time signals
        adjusted_eta = eta_pred * traffic_factor * weather_factor
        return adjusted_eta


# -------------------------------
# 5. Example Usage
# -------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = DeliveryPipeline()

    # Fake historical sequence (batch=1, seq_len=5, features=10)
    seq_data = torch.randn(1, 5, 10).to(device)

    # Fake route graph (4 nodes, 8-dim features, edges as COO index)
    graph_x = torch.randn(4, 8).to(device)
    edge_index = torch.tensor([[0, 1, 2, 3],
                               [1, 0, 3, 2]], dtype=torch.long).to(device)

    # Predict ETA distribution for node 2
    eta_pred = pipeline.predict_eta_distribution(seq_data, graph_x, edge_index, node_id=2)
    print("Predicted ETA (hours):", eta_pred.item())

    # Trust model training (features: seller history, courier stats, etc.)
    features = np.random.rand(100, 6)  # e.g., [on-time %, cancels, distance, ETA, courier rating, product type]
    labels = np.random.randint(0, 2, size=100)  # 0 = Low Trust, 1 = High Trust
    pipeline.train_trust_model(features, labels)

    trust_score = pipeline.predict_trust_score(features[:5])
    print("Trust Scores:", trust_score)

    # Real-time update
    adjusted_eta = pipeline.update_with_realtime_data(eta_pred, traffic_factor=1.2, weather_factor=1.1)
    print("Adjusted ETA with real-time data:", adjusted_eta.item())
