import torch
from torch import nn, Tensor
from torch.nn.functional import relu
from torch.nn.functional import softmax
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool
from gflownet.utils import log_memory_usage
import torch.nn.functional as F
import gc
from typing import List

from typing import Tuple

class BasePolicy(nn.Module):
    def __init__(self, node_features: int, hidden_dim: int):
        super(BasePolicy, self).__init__()
        self.node_features = node_features
        self.hid = hidden_dim
        self.in_head = 4
        self.out_head = 1
        self.gat1 = GATv2Conv(node_features, self.hid, edge_dim=1, heads=self.in_head)



class ForwardPolicy(BasePolicy):
    def __init__(self, node_features: int, hidden_dim: int, max_num_actions: int):
        super().__init__(node_features, hidden_dim)
        #log_memory_usage("Before Defining GAT2")
        self.gat2 = GATv2Conv(self.hid * self.in_head, self.hid, edge_dim=1, heads=self.out_head)
        self.fc = nn.Linear(self.hid, max_num_actions)
        #Set up FC layer for alpha
        self.alpha = torch.nn.Parameter(torch.tensor(0.0))  # Starts with equal weighting for reward function mixing parameter
    
    def forward(self, data: Data, actions: List[int]) -> Tuple[Tensor, Tensor]:
        #log_memory_usage("Before Defining Data")
        #log_memory_usage("Before Setting Up Data")
        x, edge_index, edge_attr= data.x, data.edge_index, data.edge_attr
        #print(f"Before GATConv1 x : {x.shape}")
        #print(f"Before GATConv1 edge_index : {edge_index}")
        #print(f"Before GATConv1 edge_attr : {edge_attr}")
        #log_memory_usage("Before Defining num_actions")
        num_actions = edge_attr.size(0) + 1
        #print(f"num_actions {num_actions}")
        #log_memory_usage("Before Defining GAT2")
        #self.gat2 = GATv2Conv(self.hid * self.in_head, num_actions, edge_dim=1, heads=self.out_head).to(x.device)
        
        #print(f"GAT Layer Weights: {self.gat1.lin_l.weight}")
        #log_memory_usage("Before 1st Relu")
        x = torch.relu(self.gat1(x, edge_index, edge_attr))
        #print(f"After GATConv1 x dimensions: {x.shape}")
        #print(f"GAT1 value before Relu: {self.gat1(x, edge_index, edge_attr)}")
        #print(f"After ReLu GATConv1 x dimensions: {x.dtype}")
        #log_memory_usage("Before 2nd Relu")
        
        x = torch.relu(self.gat2(x, edge_index, edge_attr))
        #print(f"After GATConv2 x dimensions: {x.shape}")
        # Apply global mean pooling to aggregate node features
        #log_memory_usage("Before Global Mean Pooling")
        x = global_mean_pool(x, batch=torch.zeros(x.size(0), dtype=torch.long))

        num_actions = edge_attr.size(0) + 1
        #print(f"Global Mean Pooling x dimensions: {x}")
        x = self.fc(x)
        x = x[:, :num_actions]
        if isinstance(actions, list):
            actions = torch.tensor(actions, dtype=torch.long) 
            
        if actions.numel() > 0:
            mask = torch.ones_like(x, dtype=torch.bool)
            mask[:, actions] = 0

            x = x.masked_fill(~mask, float('-inf'))
        #log_memory_usage("Before Softmax")
        alpha_processed = torch.sigmoid(self.alpha)
        gc.collect() 
        return torch.softmax(x, dim=1), alpha_processed

class BackwardPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, max_num_actions: int):
        super(BackwardPolicy, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_num_actions = max_num_actions

        # Define the LSTM that will process the trajectories
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Define a fully connected layer with the max number of actions
        self.fc = nn.Linear(self.hidden_dim, max_num_actions)

    def forward(self, trajectories: Tensor) -> Tensor:
        batch_size = trajectories.size(0)  # Number of trajectories
        max_len = trajectories.size(1)  # Max length of trajectories

        outputs = []

        for i in range(batch_size):
            # Get the current trajectory and mask for valid actions
            current_trajectory = trajectories[i, :].unsqueeze(0)  # Add batch dimension
            valid_mask = (current_trajectory != -1)
            num_valid_actions = valid_mask.sum().item()

            # Prepare the LSTM input (handling padded sequences)
            current_lengths = valid_mask.sum(dim=1)
            packed_input = nn.utils.rnn.pack_padded_sequence(
                current_trajectory.float().unsqueeze(-1), 
                current_lengths, 
                batch_first=True, 
                enforce_sorted=False
            )
            packed_output, _ = self.lstm(packed_input)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

            # Use the mask to gather the output from the last valid time step
            idx = (current_lengths - 1).view(-1, 1).expand(len(current_lengths), lstm_out.size(2))
            idx = idx.unsqueeze(1)
            lstm_out = lstm_out.gather(1, idx).squeeze(1)

            # Pass through the fully connected layer
            output = self.fc(lstm_out)

            # Slice off the unnecessary actions
            output = output[:, :num_valid_actions]

            # Apply softmax for probability distribution over actions
            output = torch.softmax(output, dim=1)

            # Pad the output to match the maximum number of actions (if needed)
            padded_output = F.pad(output, (0, max_len - output.size(1)), value=1)
            outputs.append(padded_output)

        outputs = torch.stack(outputs, dim=0)
        return outputs

'''        
class BackwardPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(BackwardPolicy, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = None  # This will be defined dynamically based on the number of actions.

    def forward(self, trajectories: Tensor) -> Tensor:
        batch_size = trajectories.size(0)  # Number of trajectories
        max_len = trajectories.size(1)  # Max length of trajectories

        # Initialize lists to hold outputs and masks
        outputs = []
        masks = []

        for i in range(batch_size):
            # Get the current trajectory and mask for valid actions
            current_trajectory = trajectories[i, :].unsqueeze(0) # Add batch dimension
            #print(f"current_trajectory {current_trajectory.shape}")
            valid_mask = (current_trajectory != -1)
            #print(f"valid_mask back policy {valid_mask.shape}")
            num_valid_actions = valid_mask.sum().item()
            if self.fc is None or self.fc.out_features != num_valid_actions:
            # Define the FC layer dynamically based on the number of valid actions
                self.fc = nn.Linear(self.hidden_dim, num_valid_actions).to(trajectories.device)

            # Prepare the LSTM input
            current_lengths = valid_mask.sum(dim=1)
            packed_input = nn.utils.rnn.pack_padded_sequence(current_trajectory.float().unsqueeze(-1), current_lengths, batch_first=True, enforce_sorted=False)
            packed_output, _ = self.lstm(packed_input)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

            # Use the mask to gather the output from the last valid time step
            idx = (current_lengths - 1).view(-1, 1).expand(len(current_lengths), lstm_out.size(2))
            idx = idx.unsqueeze(1)
            lstm_out = lstm_out.gather(1, idx).squeeze(1)

            # Pass through the fully connected layer and apply softmax
            output = torch.softmax(self.fc(lstm_out), dim=1)

            # Pad the output to match the maximum number of non-terminating actions
            padded_output = F.pad(output, (0, max_len - output.size(1)), value=1)
            outputs.append(padded_output)

            # Create a mask for valid positions
            #padded_mask = F.pad(valid_mask.float(), (0, max_len - valid_mask.size(1)), value=0)
            #masks.append(padded_mask)

        # Stack the outputs and masks to form tensors
        outputs = torch.stack(outputs, dim=0)
        #masks = torch.stack(masks, dim=0)
        gc.collect()
        return outputs
'''
