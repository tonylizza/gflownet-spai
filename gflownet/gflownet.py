import torch
from torch import nn
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.distributions import Categorical
from torch_geometric.data import Data
from .log import Log
from typing import List

class GFlowNet(nn.Module):
    def __init__(self, forward_policy, backward_policy, env):
        super().__init__()
        #self.total_flow = Parameter(torch.ones(1))
        self.register_buffer('total_flow', torch.ones(1))
        self.forward_policy = forward_policy
        self.backward_policy = backward_policy
        self.env = env
    
    def mask_and_normalize(self, s, probs):
        mask = self.env.mask(s)
        print(f"Original mask shape: {mask.shape}")  # Check the mask shape
        
        print(f"probs before normalization: {probs}")

        mask = mask.unsqueeze(1)
            
        #print(f"Expanded mask shape: {mask.shape}")  # Check the expanded mask shape

        probs = mask * probs
        print(f"Probs with mask applied: {probs}")  # Check masked probabilities

        # Check if there is only one row
        if probs.size(0) == 1:
            summed_probs = probs
        else:
            summed_probs = probs.sum(2)

        print(f"Summed probs: {summed_probs}")  # Check sum of probabilities

        # Ensure no division by zero
        summed_probs[summed_probs == 0] = 1

        if probs.size(0) == 1:
            normalized_probs = probs
        else:
            normalized_probs = probs / summed_probs.unsqueeze(1)

        print(f"Normalized probs: {normalized_probs.shape}")  # Check normalized probabilities

        return normalized_probs
    
    def forward_probs(self, s, actions):
        """
        Returns a vector of probabilities over actions in a given state.
        
        Args:
            s: An NxD matrix representing N states
            actions: A list containing the actions for all trajectory samples to this point.
        """
        print("Forward Probs Logging")
        print(f"Len of S for iteration: {len(s)}")

        data_list = self.state_to_data(s)
        all_probs = []
        alphas = []
        
        for data in data_list:
            probs, alpha = self.forward_policy(data)
            print(f"Probs from Policy Shape: {probs.shape}")
            all_probs.append(probs)
            alphas.append(alpha)
        # Cat probabilities on dim 0 to restore N rows from the creation of the list.
        #probs = torch.cat(all_probs, dim=0)

        init_mask_for_s = torch.stack([self.env.init_mask for _ in range(len(s))])
        probs = torch.stack(all_probs)
        mask = torch.ones_like(probs)
        if actions:
            actions = torch.stack(actions, dim=0).t()
            print(f"actions shape for mask in forward_probs: {actions.shape}")
            # Ensure actions tensor is 2D for advanced indexing
            if actions.dim() == 1:
                actions = actions.unsqueeze(1)

            for i in range(actions.shape[0]):
                for j in range(actions.shape[1]):
                    mask[i, 0, actions[i, j]] = 0
                    #if actions[i, j] < probs.size(1):  # Check if the action index is within bounds
                        #mask[i, 0, actions[i, j]] = 0  # Assuming actions are single values in a column vector
        else:
            pass

        mask = init_mask_for_s * mask
        probs = probs * mask
        print(f"mask for probs :{mask} ")
        alpha = torch.stack(alphas).mean()  # Assuming you want to average the alpha values
        
        return self.mask_and_normalize(s, probs), alpha
    
    def sample_states(self, s0, return_log=False):
        s = [matrix.clone() for matrix in s0]

        done = torch.BoolTensor([False] * len(s))
        cumulative_actions = [[] for _ in range(len(s))]
        log = Log(s0, self.backward_policy, self.total_flow, self.env) if return_log else None
        done_iterations = 0

        while not done.all():
            done_iterations += 1

            #Generate actions for all samples for logging
            print(f"Type for _actions: {type(log._actions)}")
            probs_all, _ = self.forward_probs(s, log._actions)
            print(f"Probs from Policy: {probs_all}")
            actions_all = Categorical(probs_all).sample()

            if actions_all.dim() == 0:
                actions_all = actions_all.unsqueeze(0)

            # Generate actions only for active samples
            active_indices = (~done).nonzero(as_tuple=True)[0]
            active_states = [s[i] for i in active_indices]
            actions_active = actions_all[active_indices]
            #probs, _ = self.forward_probs(active_states)

            
            for idx, action in zip(active_indices, actions_active):
                cumulative_actions[idx].append(action.item())


            # Update the environment for active samples only
            updated_matrices = self.env.update(active_states, actions_active.unsqueeze(1))
            
            for idx, update in zip(active_indices, updated_matrices):
                #print(f"S[idx] before update from env: {s[idx].shape}")
                s[idx] = update
                #print(f"S[idx] after update from env: {s[idx].shape}")

            if return_log:
                log.log(s, probs_all, actions_all, done)

            # Identify terminating actions and update the done tensor
            terminated = (actions_active == (probs_all.shape[-1] - 1)).view(-1)  # Ensure terminated is a 1D tensor
            print(f"Terminated Shape: {terminated}")
            done[active_indices] = terminated
            if done_iterations % 1000 == 0:
                print(f"sample states s shape: {len(s)}")
                print(f"sample states done shape: {done.shape}")
                print(f"sample states actions shape: {actions_active.shape}")
                print(f"Sampling Iterations {done_iterations}")
                print(f"Updated done tensor: {done}")
                print(f"Probs shape Sample State {probs_all.shape}")

        return (s, log) if return_log else s
    
    def evaluate_trajectories(self, traj, actions):
        num_samples = len(traj)
        traj = traj.reshape(-1, traj.shape[-1])
        actions = actions.flatten()
        finals = traj[actions == self.env.num_actions - 1]
        zero_to_n = torch.arange(len(actions))
        
        fwd_probs, _ = self.forward_probs(traj)
        fwd_probs = torch.where(actions == -1, 1, fwd_probs[zero_to_n, actions])
        fwd_probs = fwd_probs.reshape(num_samples, -1)
        
        actions = actions.reshape(num_samples, -1)[:, :-1].flatten()
        
        back_probs = self.backward_policy(traj)
        back_probs = back_probs.reshape(num_samples, -1, back_probs.shape[1])
        back_probs = back_probs[:, 1:, :].reshape(-1, back_probs.shape[2])
        back_probs = torch.where((actions == -1) | (actions == 2), 1,
                                 back_probs[zero_to_n[:-num_samples], actions])
        back_probs = back_probs.reshape(num_samples, -1)
        
        rewards = self.env.reward(finals)
        
        return fwd_probs, back_probs, rewards

    def state_to_data(self, s: List[Tensor]) -> list:
        """
        Converts a list of sparse tensors to a list of PyTorch Geometric Data objects.
        
        Args:
            s: A list of sparse tensors representing N states
        
        Returns:
            data_list: A list of PyTorch Geometric Data objects
        """
        batch_size = len(s)
        data_list = []
        
        for i, current_matrix in enumerate(s):
            #print(f"Current Matrix Size Comparison {i}: {current_matrix.size()}")
            if not current_matrix.is_sparse:
                raise ValueError(f"Tensor at index {i} is not a sparse tensor.")
            
            current_matrix = s[i]
            #print(f"Current Matrix State to Data: {current_matrix}")
            edge_index = current_matrix._indices()
            edge_attr = current_matrix._values()
            num_nodes = current_matrix.size(0)
            #x = torch.ones((num_nodes, 1))  # Example node features, you may adjust this as needed
            x = torch.ones((324, 1))
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr.float())
            data_list.append(data)
            
            print(f"Length of data_list: {len(data_list)}")
            print(f"Data object {i} - x dimensions: {data.x.shape}")
            #print(f"Data object {i} - x values: {data.x}")
            #print(f"Data object {i} - edge_index dimensions: {data.edge_index}")
            #print(f"Data object {i} - edge_attr dimensions: {data.edge_attr}")
        
        return data_list

