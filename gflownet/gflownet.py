import torch
from torch import nn
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.distributions import Categorical
from torch_geometric.data import Data
from .log import Log
from .utils import resize_sparse_tensor, log_memory_usage, malloc_usage
from typing import List
import gc

class GFlowNet(nn.Module):
    def __init__(self, forward_policy, backward_policy, env):
        super().__init__()
        #self.total_flow = Parameter(torch.ones(1))
        self.register_buffer('total_flow', torch.ones(1))
        self.forward_policy = forward_policy
        self.backward_policy = backward_policy
        self.env = env
    
    def mask_and_normalize(self, s, probs):
        #log_memory_usage("mask_and_normalize: Before getting self.env.mask")
        #mask = self.env.mask(s)
        #print(f"Original mask shape: {mask.shape}")  # Check the mask shape
        
        #print(f"probs before normalization: {probs}")

        #mask = mask.unsqueeze(1)
            
        #print(f"Expanded mask shape: {mask.shape}")  # Check the expanded mask shape

        #probs.mul_(mask)
        #log_memory_usage("mask_and_normalize: After prob.mul_(mask)")
        #print(f"Probs with mask applied: {probs.shape}")  # Check masked probabilities
        #del mask
        # Check if there is only one row
        if probs.size(0) > 1:
            summed_probs = probs.sum(2)
            summed_probs[summed_probs == 0] = 1  # Avoid division by zero
            probs.div_(summed_probs.unsqueeze(1))  # In-place division for normalization
        # If only one row, probs remain unchanged.
        
        del summed_probs
        gc.collect()
        return probs
    
    def forward_probs(self, s, data_list, actions=None):
        """
        Returns a vector of probabilities over actions in a given state.
        
        Args:
            s: An NxD matrix representing N states
            actions: A list containing the actions for all trajectory samples to this point.
        """
        #print("Forward Probs Logging")
        #print(f"Len of S for iteration: {len(s)}")
        #log_memory_usage("Before Converting States to Data")
        #log_memory_usage("After Converting States to Data")
        all_probs = []
        alphas = []
        
        # Handle empty actions
        if actions is None or len(actions) == 0:
            actions = torch.empty(0)
        else:
            # Transpose and convert to tensor
            actions = torch.tensor(list(zip(*actions)), dtype=torch.long)

        #print(f"actions type {type(actions)}")
        for i, data in enumerate(data_list):
            #print(f"Data for Forward Policy {data}")
            #log_memory_usage("Before Forward Policy Sampling")
            actions_data = actions[i, :] if actions.numel() > 0 else torch.empty(0, dtype=torch.long)
            probs, alpha = self.forward_policy(data, actions_data)
            #log_memory_usage("After Forward Policy")
            #print(f"Probs from Policy Shape {probs.shape}")
            #print(f"Probs from Policy: {probs}")
            all_probs.append(probs)
            alphas.append(alpha)
            del data, probs, alpha
            gc.collect()
        # Cat probabilities on dim 0 to restore N rows from the creation of the list.
        #probs = torch.cat(all_probs, dim=0)

        #init_mask_for_s = torch.stack([self.env.init_mask for _ in range(len(s))])
        #print(f"init_mask_for_s: {init_mask_for_s.shape}")
        #probs = torch.stack(all_probs)
        all_probs = torch.stack(all_probs, dim=0)
        alpha = torch.stack(alphas, dim=0).mean()
        mask = torch.ones_like(all_probs)
        #print(f"mask {mask.shape}")
        #log_memory_usage("forward_probs: Before removing actions")
        '''
        if actions:
            actions = torch.stack(actions, dim=0).t()
            #print(f"actions shape for mask in forward_probs: {actions.shape}")
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
        '''
        #mask = init_mask_for_s * mask
        all_probs.mul_(mask)
        del mask, data_list
        gc.collect()
        #log_memory_usage("forward_probs: Before passing to mask_and_normalize")
        #print(f"mask for probs :{mask} ")

        if all_probs.size(0) > 1:
            summed_probs = all_probs.sum(2)
            summed_probs[summed_probs == 0] = 1  # Avoid division by zero
            all_probs.div_(summed_probs.unsqueeze(1))  # In-place division for normalization
        # If only one row, probs remain unchanged.
        del summed_probs
        #return self.mask_and_normalize(s, all_probs), alpha
        return all_probs, alpha
    
    def sample_states(self, s0, return_log=False):
        #s = [matrix.clone() for matrix in s0]

        done = torch.BoolTensor([False] * len(s0))
        log = Log(s0, self.backward_policy, self.total_flow, self.env) if return_log else None
        done_iterations = 0
        #print(f"Begin sampling")

        data_list = self.state_to_data(s0)

        while not done.all():
            done_iterations += 1

            if done_iterations % 1000 == 0:
                log_memory_usage("Sample Iteration " + str(done_iterations))
            else:
                pass
            #Generate actions for all samples for logging
            #print(f"Type for _actions: {type(log._actions)}")
            #log_memory_usage("Before Forward Probs")
            probs_all, _ = self.forward_probs(s0, data_list, log._actions)
            #print(f"probs_all {probs_all.shape}")
            #log_memory_usage("After Forward Probs Created")
            actions_all = Categorical(probs_all).sample()
            #print(f"actions_all {actions_all.shape}")
            #log_memory_usage("After Categorical Sampling")

            if actions_all.dim() == 0:
                actions_all = actions_all.unsqueeze(0)

            # Generate actions only for active samples
            active_indices = (~done).nonzero(as_tuple=True)[0]
            #active_states = [s[i] for i in active_indices]
            #print(f"Active States {active_states}")
            actions_active = actions_all[active_indices]
            #print(f"Actions Active {actions_active}")
            #probs, _ = self.forward_probs(active_states)

            
            '''
            # Update the environment for active samples only
            updated_matrices = self.env.update(active_states, actions_active.unsqueeze(1))
            
            for idx, update in zip(active_indices, updated_matrices):
                #print(f"S[idx] before update from env: {s[idx].shape}")
                s[idx] = update
                #print(f"S[idx] {idx} NNZ: {s[idx]._nnz()}")
            '''
            if return_log:
                log.log(s0, probs_all, actions_all, done)

            # Identify terminating actions and update the done tensor
            terminated = (actions_active == (probs_all.shape[-1] - 1)).view(-1)  # Ensure terminated is a 1D tensor
            #print(f"Terminated Shape: {terminated}")
            done[active_indices] = terminated

        complete_actions = log.actions.t()
        #print(f"Complete Actions {complete_actions.shape}")
        reduced_matrices = self.env.update(s0, complete_actions)
        reward_matrices = [resize_sparse_tensor(reduced_matrices[i], (self.env.matrix_size, self.env.matrix_size)) for i in range(len(reduced_matrices))]
        rewards = torch.tensor([self.env.reward(matrix, len(log._actions), self.forward_policy.alpha) for matrix in reward_matrices], dtype=log.rewards.dtype)
        log.rewards = rewards
        del reduced_matrices
        del reward_matrices
        gc.collect()
        return log if return_log else None
    
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
            

            edge_index = s[i]._indices()
            edge_attr = s[i]._values()

            #print(f"Current Matrix State to Data: {current_matrix}")
            #x = torch.ones((num_nodes, 1))  # Example node features, you may adjust this as needed
            x = torch.ones((self.env.matrix_size*2, 1))
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr.float())
            data_list.append(data)
            
            #print(f"Length of data_list: {len(data_list)}")
            #print(f"Data object {i} - x dimensions: {data.x.shape}")
            #print(f"Data object {i} - x values: {data.x}")
            #print(f"Data object {i} - edge_index dimensions: {data.edge_index}")
            #print(f"Data object {i} - edge_attr dimensions: {data.edge_attr}")
        
        return data_list

