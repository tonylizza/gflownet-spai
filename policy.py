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
        self.potential_head = nn.Linear(self.hid, 1)
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
        #print(f"X Global Mean shape {x.shape}")
        state_potential = torch.relu(self.potential_head(x).squeeze(-1))
        #Compute state potential
        #print(f"State potential requires grad: {state_potential.requires_grad}")

        #num_actions = edge_attr.size(0) + 1
        #Create a diagonal mask to prevent removal of diagonal elements (required for ILU)
        diagonal_mask = edge_index[0] != edge_index[1]
        diagonal_mask = torch.cat([diagonal_mask, torch.tensor([True])])

        #print(f"Global Mean Pooling x dimensions: {x}")
        x = self.fc(x)
        x = x[:, :num_actions]
        if isinstance(actions, list):
            actions = torch.tensor(actions, dtype=torch.long) 
            
        if actions.numel() > 0:
            action_mask = torch.ones_like(x, dtype=torch.bool)
            action_mask[:, actions] = 0

            mask = action_mask & diagonal_mask
        
        else:
            mask = diagonal_mask

            
        x = x.masked_fill(~mask, float('-inf'))
        #log_memory_usage("Before Softmax")
        alpha_processed = torch.sigmoid(self.alpha)
        gc.collect() 
        return torch.softmax(x, dim=1), state_potential, alpha_processed
    

class BackwardPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, max_num_actions: int):
        super(BackwardPolicy, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_num_actions = max_num_actions

        # Define the LSTM that will process the trajectories
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Define a fully connected layer with the max number of actions
        self.fc = nn.Linear(self.hidden_dim, max_num_actions)

        # Define a layer for computing state potentials
        self.potential_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, trajectories: torch.Tensor, valid_actions_list: List[List[int]], termination_action_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
        - trajectories: Tensor of shape (batch_size, max_len)
        - valid_actions_list: A list where each element is a list of valid action indices for that sample
        - termination_action_index: The index corresponding to the termination action
        """
        batch_size, max_len = trajectories.size()

        # Create a mask for valid positions (non-padded)
        valid_seq_mask = (trajectories != -1)  # Shape: (batch_size, max_len)

        # Compute lengths of sequences
        sequence_lengths = valid_seq_mask.sum(dim=1).long()  # Shape: (batch_size)

        # Prepare inputs for the LSTM
        inputs = trajectories.float().unsqueeze(-1)  # Shape: (batch_size, max_len, 1)

        # Pack the sequences for the LSTM
        packed_input = nn.utils.rnn.pack_padded_sequence(
            inputs,
            sequence_lengths.cpu(),  # Ensure lengths are on CPU
            batch_first=True,
            enforce_sorted=False
        )

        # Run the LSTM
        packed_output, _ = self.lstm(packed_input)

        # Unpack the outputs
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=max_len  # Ensure the output has the correct length
        )  # Shape: (batch_size, max_len, hidden_dim)

        # Compute state potentials
        potentials = self.potential_head(lstm_out).squeeze(-1)  # Shape: (batch_size, max_len)

        # Pass through the fully connected layer to get logits for actions
        logits = self.fc(lstm_out)  # Shape: (batch_size, max_len, max_num_actions)

        # Clip logits to the number of valid actions including the termination action
        logits = logits[:, :, :termination_action_index]  # Shape: (batch_size, max_len, termination_action_index)

        # Initialize action masks to zeros (this mask will have the same size as the logits)
        action_masks = torch.zeros(batch_size, max_len, termination_action_index)

        # For padded positions, only the termination action is valid
        padded_positions = (trajectories == -1)  # Shape: (batch_size, max_len)
        padded_positions_indices = padded_positions.nonzero(as_tuple=False)  # Shape: (num_padded_positions, 2)

        if padded_positions_indices.numel() > 0:
            batch_indices = padded_positions_indices[:, 0]  # Shape: (num_padded_positions,)
            time_indices = padded_positions_indices[:, 1]   # Shape: (num_padded_positions,)
            # Set the termination action probability to 1 at padded positions
            action_masks[batch_indices, time_indices, termination_action_index - 1] = 1

        # For valid positions, set valid actions
        for i in range(batch_size):
            valid_actions = valid_actions_list[i]  # Valid actions for sample i (excluding -1)
            
            # Ensure the sample_mask has the right size to accommodate the termination action
            sample_mask_size = termination_action_index  
            sample_mask = torch.zeros(sample_mask_size)  # Mask is of size 209 to accommodate index 208

            # Now we can safely index into sample_mask with valid_actions
            sample_mask[valid_actions] = 1  # Set valid actions to 1

            # Apply the mask to valid positions in the sequence
            valid_positions = valid_seq_mask[i]  # Shape: (max_len)

            # Ensure the mask matches the number of actions expected
            action_masks[i, valid_positions, :] = sample_mask  # Apply mask up to termination action

        # Apply the action masks to the logits (invalid actions get -inf)
        masked_logits = logits.masked_fill(action_masks == 0, float('-inf'))

        # Compute softmax over the valid actions
        output = torch.softmax(masked_logits, dim=2).unsqueeze(-2)  # Shape: (batch_size, max_len, 1, termination_action_index)

        return output, potentials  # Final output with shape (batch_size, max_len, termination_action_index)




'''
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