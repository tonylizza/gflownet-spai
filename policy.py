import torch
from torch import nn, Tensor
from torch.nn.functional import relu
from torch.nn.functional import softmax
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool
from gflownet.utils import log_memory_usage
import torch.nn.functional as F
import gc

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
    def __init__(self, node_features: int, hidden_dim: int):
        super().__init__(node_features, hidden_dim)
        #log_memory_usage("Before Defining GAT2")
        self.gat2 = None #Define dynamically at runtime
        self.alpha = torch.nn.Parameter(torch.tensor(0.0))  # Starts with equal weighting for reward function mixing parameter
    
    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        #log_memory_usage("Before Defining Data")
        log_memory_usage("Before Setting Up Data")
        x, edge_index, edge_attr= data.x, data.edge_index, data.edge_attr
        #print(f"Before GATConv1 x : {x}")
        #print(f"Before GATConv1 edge_index : {edge_index}")
        #print(f"Before GATConv1 edge_attr : {edge_attr}")
        log_memory_usage("Before Defining num_actions")
        num_actions = edge_attr.size(0) + 1
        #print(f"num_actions {num_actions}")
        #log_memory_usage("Before Defining GAT2")
        if self.gat2 is None or self.gat2.out_channels != num_actions:
            if self.gat2 is not None:
                del self.gat2
                gc.collect()
            self.gat2 = GATv2Conv(self.hid * self.in_head, num_actions, edge_dim=1, heads=self.out_head).to(x.device)
        
        #print(f"GAT Layer Weights: {self.gat1.lin_l.weight}")
        log_memory_usage("Before 1st Relu")
        x = torch.relu(self.gat1(x, edge_index, edge_attr))
        print(f"After GATConv1 x dimensions: {x.shape}")
        #print(f"GAT1 value before Relu: {self.gat1(x, edge_index, edge_attr)}")
        #print(f"After ReLu GATConv1 x dimensions: {x.dtype}")
        log_memory_usage("Before 2nd Relu")
        
        x = torch.relu(self.gat2(x, edge_index, edge_attr))
        print(f"After GATConv2 x dimensions: {x.shape}")
        # Apply global mean pooling to aggregate node features
        log_memory_usage("Before Global Mean Pooling")
        x = global_mean_pool(x, batch=torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        #print(f"Global Mean Pooling x dimensions: {x}")
        log_memory_usage("Before Softmax")
        gc.collect() 
        return torch.softmax(x, dim=1), torch.sigmoid(self.alpha)

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
class BackwardPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(BackwardPolicy, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = None

    def forward(self, trajectories: Tensor) -> Tensor:
        #Create num_actions dynamically based on input
        #log_memory_usage("Before Defining num_actions")
        num_actions = (trajectories != -1).sum().item() + 1
        print(f"Num_actions back policy: {num_actions}")
        #log_memory_usage("Before Defining Fully Connected Layer")
        #Define the FC layer dynamically based on num_actions
        self.fc = nn.Linear(self.hidden_dim, num_actions).to(trajectories.device) 
        #log_memory_usage("Before Creating Mask")
        # Create a mask based on the padding value (-1)
        trajectories = trajectories.float().unsqueeze(-1)
        mask = (trajectories != -1)
        #print(f"mask length {mask.shape}")
        lengths = mask.sum(dim=1).squeeze()
        print(f"Length of Back Prob {lengths.shape}")
        #log_memory_usage("Before padding sequences")
        # Pack the padded sequences
        packed_input = nn.utils.rnn.pack_padded_sequence(trajectories, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        #log_memory_usage("Before Gathering Output")
        # Use the mask to gather the output from the last valid time step
        idx = (lengths - 1).view(-1, 1).expand(len(lengths), lstm_out.size(2))
        time_dimension = 1
        idx = idx.unsqueeze(time_dimension)
        #log_memory_usage("Before LSTM")
        lstm_out = lstm_out.gather(time_dimension, idx).squeeze(time_dimension)
        
        output = self.fc(lstm_out)
        return torch.softmax(output, dim=1)
'''