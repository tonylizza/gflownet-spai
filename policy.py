import torch
from torch import nn, Tensor
from torch.nn.functional import relu
from torch.nn.functional import softmax
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool

from typing import Tuple

'''
#Forward Policy uses a graph attention network
class ForwardPolicy(nn.Module):
    def __init__(self, node_features: int, hidden_dim: int, num_actions: int):
        super().__init__()
        self.gat1 = GATConv(node_features, hidden_dim)
        self.gat2 = GATConv(hidden_dim, num_actions)
        self.alpha = torch.nn.Parameter(torch.tensor(0.0)) #Starts with equal weighting for reward function mixing parameter
    
    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        x, edge_index = data.x, data.edge_index
        print(f"Input x dimensions: {x.shape}")
        print(f"Input edge_index dimensions: {edge_index.shape}")
        x = torch.relu(self.gat1(x, edge_index))
        print(f"After GATConv1 x dimensions: {x.shape}")
        x = self.gat2(x, edge_index)
        print(f"After GATConv2 x dimensions: {x.shape}")
        return softmax(x, dim=1), torch.sigmoid(self.alpha)
'''
class ForwardPolicy(nn.Module):
    def __init__(self, node_features: int, hidden_dim: int, num_actions: int):
        super().__init__()
        self.hid = hidden_dim
        self.in_head = 8
        self.out_head = 1
        self.gat1 = GATv2Conv(node_features, self.hid, edge_dim=1, heads=self.in_head)
        self.gat2 = GATv2Conv(self.hid * self.in_head, num_actions, edge_dim=1, heads=self.out_head)
        #self.fc = nn.Linear(num_actions - 1, num_actions)
        #self.lin = torch.nn.Linear(hidden_dim, num_actions)
        self.alpha = torch.nn.Parameter(torch.tensor(0.0))  # Starts with equal weighting for reward function mixing parameter

        # Initialize parameters
        #self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            #print(f"Module: {m}")
            if isinstance(m, GATv2Conv):
                #if m.lin_r.bias is not None:
                torch.nn.init.constant_(m.lin_r.bias, 0.0)
                #print(f"Initialized lin_r.bias as {m.lin_r.bias}")
                #if m.lin_r.weight is not None:
                torch.nn.init.xavier_uniform_(m.lin_r.weight)
                #print(f"Initialized lin_r.weight as {m.lin_r.weight}")
                #if m.att is not None:
                torch.nn.init.xavier_uniform_(m.att)
                #print(f"Initialized m.att weight as {m.att}")
    
    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        x, edge_index, edge_attr= data.x, data.edge_index, data.edge_attr
        #print(f"Input x dimensions: {x.shape}")
        #print(f"Input edge_index dimensions: {edge_index.shape}")
        
        x = torch.relu(self.gat1(x, edge_index, edge_attr))
        #print(f"After GATConv1 x dimensions: {x.shape}")
        
        x = torch.relu(self.gat2(x, edge_index, edge_attr))
        # Apply global mean pooling to aggregate node features
        x = global_mean_pool(x, batch=torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        
        #x = self.lin(x)
        #Map GAT2 back to fully connected layer to get to [1,325]
        #x = self.fc(x)
        #print(f"After GATConv2 x dimensions: {x.shape}")
        
        return torch.softmax(x, dim=1), torch.sigmoid(self.alpha)


            
class BackwardPolicy:
    def __init__(self, matrix_size, num_actions):
        self.matrix_size = matrix_size
        self.num_actions = num_actions
    
    def __call__(self, s, env, terminated):
        # s: current state tensor
        # env: the environment object with current matrix state
        #Policy code needs to be changed to send back back probs for a collection of samples and idx needs to be fixed. Also, give these variables more descriptive names than just s.

        batch_size = s.shape[0]
        #print(f"S for back policy: {len(s)}")
        # Determine the state index
        idx = s.argmax(dim=-1)
        #print(f"idx for back policy: {idx.shape}")
        # Initialize probabilities
        probs = torch.ones(batch_size, idx.shape[1]) * 0.5
        #print(f"probs shape for back policy: {probs.shape}")
        # Calculate the probability of transitioning to previous states
        for i in range(batch_size):

            # If idx[i] has more than one element, reshape or reduce it
            # Assuming idx[i] should be a single value or we should select one value
            if idx[i].numel() > 1:
                # Flatten and get the first element (modify this as needed for your logic)
                idx_i = idx[i].view(-1)[0].item()
            else:
                idx_i = idx[i].item()

            #row, col = divmod(idx[i].item(), self.matrix_size)
            row, col = divmod(idx_i, self.matrix_size)
            if (row, col, 0) in env.removed_entries:
                # Higher probability if the entry was previously removed. Fix this later to have realistic values.
                #probs[i, idx[i]] = 0.8
                probs[i, idx_i] = 0.8
        
        # Set probabilities for edge cases
        #at_top_edge = idx < self.matrix_size
        #at_left_edge = (idx > 0) & (idx % self.matrix_size == 0)
        #probs[at_left_edge] = torch.Tensor([1, 0, 0])
        #probs[at_top_edge] = torch.Tensor([0, 1, 0])

        #Count the number of non-terminating states for each row.
        non_terminating = ~terminated

        #print(f"non_terminating.shape {non_terminating.shape}")

        num_samples = torch.arange(batch_size)
        #print(f"num_samples {num_samples}")
        num_true_in_specific_row = torch.sum(non_terminating[num_samples], dim=1)
        #print(f"num_true_in_specific_row {num_true_in_specific_row}")

        for i in range(batch_size):
            probs[i] = torch.where(terminated[i], 1, torch.tensor(1.0/num_true_in_specific_row[i], dtype=torch.float64))
        #probs[:, -1] = 0  # Disregard termination
        
        # Normalize probabilities
        #probs = probs / probs.sum(dim=1, keepdim=True)

        
        
        #print(f"final probs: {probs}")
        
        return probs