import torch
from torch import nn, Tensor
from torch.nn.functional import relu
from torch.nn.functional import softmax
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool

from typing import Tuple

class BasePolicy(nn.Module):
    def __init__(self, node_features: int, hidden_dim: int):
        super(BasePolicy, self).__init__()
        self.node_features = node_features
        self.hid = hidden_dim
        self.in_head = 8
        self.out_head = 1
        self.gat1 = GATv2Conv(node_features, self.hid, edge_dim=1, heads=self.in_head)



class ForwardPolicy(BasePolicy):
    def __init__(self, node_features: int, hidden_dim: int, num_actions: int):
        super().__init__(node_features, hidden_dim)
        self.num_actions = num_actions
        self.gat2 = GATv2Conv(self.hid * self.in_head, self.num_actions, edge_dim=1, heads=self.out_head)
        self.alpha = torch.nn.Parameter(torch.tensor(0.0))  # Starts with equal weighting for reward function mixing parameter
    
    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        x, edge_index, edge_attr= data.x, data.edge_index, data.edge_attr
        #print(f"Before GATConv1 x : {x}")
        #print(f"Before GATConv1 edge_index : {edge_index}")
        #print(f"Before GATConv1 edge_attr : {edge_attr}")

        
        #print(f"GAT Layer Weights: {self.gat1.lin_l.weight}")
        x = torch.relu(self.gat1(x, edge_index, edge_attr))
        #print(f"GAT1 value before Relu: {self.gat1(x, edge_index, edge_attr)}")
        #print(f"After GATConv1 x dimensions: {x}")
        
        x = torch.relu(self.gat2(x, edge_index, edge_attr))
        #print(f"After GATConv2 x dimensions: {x}")
        # Apply global mean pooling to aggregate node features
        x = global_mean_pool(x, batch=torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        #print(f"Global Mean Pooling x dimensions: {x}") 
        return torch.softmax(x, dim=1), torch.sigmoid(self.alpha)

class BackwardPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_actions: int):
        super(BackwardPolicy, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_actions)

    def forward(self, trajectories: Tensor) -> Tensor:
        # Create a mask based on the padding value (-1)
        trajectories = trajectories.float()
        trajectories = trajectories.unsqueeze(-1)
        mask = (trajectories != -1)
        #print(f"mask length {mask.shape}")
        lengths = mask.sum(dim=1)
        lengths = lengths.squeeze()
        
        # Pack the padded sequences
        packed_input = nn.utils.rnn.pack_padded_sequence(trajectories, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Use the mask to gather the output from the last valid time step
        idx = (lengths - 1).view(-1, 1).expand(len(lengths), lstm_out.size(2))
        time_dimension = 1
        idx = idx.unsqueeze(time_dimension)
        lstm_out = lstm_out.gather(time_dimension, idx).squeeze(time_dimension)
        
        output = self.fc(lstm_out)
        return torch.softmax(output, dim=1)