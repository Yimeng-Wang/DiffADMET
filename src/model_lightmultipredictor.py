import torch
import torch.nn as nn
import torch.nn.functional as F
from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout


class GraphExtractor(nn.Module):
    def __init__(
        self,
        node_feat_size,
        edge_feat_size,
        graph_feat_size,
        num_layers=None,
        dropout=0.0,
        num_timesteps=None,
    ):
        super().__init__()
        self.gnn_drug = AttentiveFPGNN(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            num_layers=num_layers,
            graph_feat_size=graph_feat_size,
            dropout=dropout,
        )
        self.readout = AttentiveFPReadout(
            feat_size=graph_feat_size,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )

    def forward(self, bg, n_feats, e_feats):
        return self.readout(bg, self.gnn_drug(bg, n_feats, e_feats))


class LightweightFusionLayer(nn.Module):
    def __init__(
        self,
        node_feat_size,
        edge_feat_size,
        graph_feat_size,
        num_layer,
        dropout,
        num_timestep,
        predictor_hidden_feats,
    ):
        super().__init__()
        self.graph_extractor = GraphExtractor(
            node_feat_size,
            edge_feat_size,
            graph_feat_size,
            num_layer,
            dropout,
            num_timestep,
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(graph_feat_size * 2, predictor_hidden_feats),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(predictor_hidden_feats, predictor_hidden_feats),
        )

    def forward(self, graph_2d):
        feature = self.graph_extractor(*graph_2d)
        return self.fusion_layer(torch.cat([feature, feature], dim=1))


class LightweightMultiHeadAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        if feature_dim % num_heads:
            raise ValueError("feature_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        reduced_dim = feature_dim // 2
        self.qkv_proj = nn.Linear(feature_dim, reduced_dim * 3)
        self.out_proj = nn.Linear(reduced_dim, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, modality_features, task_features):
        stacked_modality = torch.stack(modality_features, dim=1)
        stacked_tasks = torch.stack(task_features, dim=1)
        query = self.qkv_proj(stacked_tasks).chunk(3, dim=-1)[0]
        key = self.qkv_proj(stacked_modality).chunk(3, dim=-1)[1]
        value = self.qkv_proj(stacked_modality).chunk(3, dim=-1)[2]
        weights = F.softmax(torch.matmul(query, key.transpose(-2, -1)) * self.scale, dim=-1)
        context = torch.matmul(self.dropout(weights), value)
        output = self.out_proj(context)
        output = self.layer_norm(output + stacked_tasks)
        return [output[:, i, :] for i in range(output.size(1))]


class MultiTaskAdaptiveTransferModel_disc(nn.Module):
    """Fixed 2D fusion, 2D/3D task attention and matrix-directed transfer."""

    def __init__(
        self,
        node_feat_sizes,
        edge_feat_sizes,
        graph_feat_size,
        num_layer,
        dropout,
        num_timestep,
        predictor_hidden_feats,
        num_tasks,
        task_correlations,
    ):
        super().__init__()
        if len(node_feat_sizes) != 2 or len(edge_feat_sizes) != 2:
            raise ValueError("Exactly two graph modalities are required: 2D and 3D")
        self.num_tasks = num_tasks
        relations = torch.as_tensor(task_correlations, dtype=torch.bool)
        if relations.shape != (num_tasks, num_tasks):
            raise ValueError("Task relationships must match the task count")
        self.register_buffer("task_correlations", relations)
        self.task_weights = nn.Parameter(torch.randn(num_tasks, num_tasks))
        with torch.no_grad():
            self.task_weights.copy_(
                relations.float() + 0.01 * torch.randn_like(self.task_weights) * relations.float()
            )
        self.graph_extractors = nn.ModuleList(
            [
                GraphExtractor(node, edge, graph_feat_size, num_layer, dropout, num_timestep)
                for node, edge in zip(node_feat_sizes, edge_feat_sizes)
            ]
        )
        self.fusion_predictor = LightweightFusionLayer(
            node_feat_sizes[0],
            edge_feat_sizes[0],
            graph_feat_size,
            num_layer,
            dropout,
            num_timestep,
            predictor_hidden_feats,
        )
        self.task_layers = nn.ModuleList(
            [nn.Linear(predictor_hidden_feats, graph_feat_size) for _ in range(num_tasks)]
        )
        self.output_layers = nn.ModuleList(
            [nn.Linear(graph_feat_size, 1) for _ in range(num_tasks)]
        )
        self.multihead_attention = LightweightMultiHeadAttention(graph_feat_size, num_heads=4)
        self.transfer_rate = nn.Parameter(torch.tensor(3.0, dtype=torch.float32))

    def get_transfer_weights(self):
        weights = self.task_weights * self.task_correlations
        column_sums = weights.sum(dim=0, keepdim=True)
        column_sums = torch.where(column_sums > 0, column_sums, torch.ones_like(column_sums))
        return weights / column_sums

    def forward(self, graph_data):
        if len(graph_data) != 2 or any(len(item) != 3 for item in graph_data):
            raise ValueError(
                "Expected [[graph_2d, nodes_2d, edges_2d], [graph_3d, nodes_3d, edges_3d]]"
            )
        modality_features = [
            extractor(*item) for extractor, item in zip(self.graph_extractors, graph_data)
        ]
        fused_features = self.fusion_predictor(graph_data[0])
        task_features = [layer(fused_features) for layer in self.task_layers]
        task_features = torch.stack(
            self.multihead_attention(modality_features, task_features), dim=1
        )
        transferred = torch.einsum("ij,bjk->bik", self.get_transfer_weights(), task_features)
        task_features = task_features + torch.sigmoid(self.transfer_rate) * (
            transferred - task_features
        )
        return torch.stack(
            [layer(task_features[:, i, :]) for i, layer in enumerate(self.output_layers)], dim=1
        ).squeeze(-1)
