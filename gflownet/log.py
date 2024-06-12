import torch
from torch import Tensor
from typing import Tuple, List

from .utils import concatenate_sparse_tensors
from .utils import SparseTensorManipulator
from .utils import resize_sparse_tensor, resize_sparse_tensor_to_flat

class Log:
    def __init__(self, s0, backward_policy, total_flow, env):
        #self._traj = [s0]
        resized_s0 = [resize_sparse_tensor_to_flat(i, (1, 324)) for i in s0]
        self._traj = [torch.stack(resized_s0)]
        self._fwd_probs = []
        self._back_probs = None
        self._actions = []
        self.rewards = torch.zeros(len(s0))
        self.backward_policy = backward_policy
        self.total_flow = total_flow
        self.env = env
        self.num_samples = len(s0)
    
    def log(self, s: List[Tensor], probs: Tensor, actions: Tensor, done: Tensor):
        """
        Logs relevant information about each sampling step
        
        Args:
            s: A list of sparse tensors containing the current state of complete and
            incomplete samples
            
            probs: An NxA matrix containing the forward probabilities output by the
            GFlowNet for the given states
            
            actions: A Nx1 vector containing the actions taken by the GFlowNet
            in the given states
            
            done: An Nx1 Boolean vector indicating which samples are complete
            (True) and which are incomplete (False)
        """
        print(f"s length: {len(s)}")
        print(f"probs log shape: {probs.shape}")
        print(f"actions shape: {actions.shape}")
        print(f"done shape: {done.shape}")
        
        just_finished = actions == probs.shape[-1] - 1
        just_finished = just_finished.view(-1)  # Flatten to 1D
        print(f"just_finished shape: {just_finished.shape}")
        print(f"Actions for mask: {actions}")
        active = ~done.clone()
    

        active_clone = active.clone()
        #print(f"Active Clone Shape: {active_clone.shape}")
        #print(f"Active Clone: {active_clone}")
        

        # Store the updated state
        #states = [s[i].clone() for i in range(len(s))]
        #self._traj.append(states)

        # Store the updated state as a stacked tensor
        states = torch.stack([s[i].to_dense() for i in range(len(s))])
        self._traj.append(states)        

        #fwd_probs = torch.ones(actions.shape[0], probs.shape[2])
        fwd_probs = torch.ones(actions.shape[0])
        #print(f"actions log shape: {actions.shape}")
        #print(f"probs new: {probs.shape}")
        gathered_probs = probs.gather(2, actions.unsqueeze(1)).squeeze()
        #print(f"gathered_probs shape: {gathered_probs.shape}")
        #print(f"fwd_probs shape: {fwd_probs.shape}")

        # Ensure the shapes match
        mask = ~done.flatten().bool()
        #print(f"mask shape: {mask.shape}")

        fwd_probs[mask] = gathered_probs[mask]
        self._fwd_probs.append(fwd_probs)

        #_actions = -torch.ones(self.num_samples, actions.shape[1]).long()
        #_actions = torch.ones(self.num_samples, actions.numel()).to(actions.dtype)
        _actions = -torch.ones(self.num_samples, dtype=actions.dtype).long()
        actions = actions.squeeze()
        _actions[mask] = actions[mask]
        self._actions.append(_actions)
        reward_indices = torch.nonzero(just_finished.view(-1)).view(-1)
        #print(f"reward_indices: {reward_indices}")
        
        if reward_indices.numel() > 0:
            reward_matrices = [resize_sparse_tensor(s[i], (self.env.matrix_size, self.env.matrix_size)) for i in reward_indices]
            #Convert to 18x18 before calculating reward
            #reward_matrices = [s[i].to_dense().view(18, 18) for i in reward_indices]
            print(f"reward_matrices shape {len(reward_matrices)}")
            print(f"reward_matrices full {reward_matrices}")
            print(f"len(self._traj) {len(self._traj)}")
            rewards = torch.tensor([self.env.reward(matrix, len(self._traj)) for matrix in reward_matrices], dtype=self.rewards.dtype)
            print(f"reward values: {rewards}")

            self.rewards[reward_indices] = rewards
        

    @property
    def traj(self):
        if isinstance(self._traj, list):
            #print(f"self._traj elements: {[type(elem) for elem in self._traj]}")
            self._traj = [tensor.to_dense() if tensor.is_sparse else tensor for tensor in self._traj]
            print(f"self._traj shapes: {[tensor.shape for tensor in self._traj]}")
            #self._traj = torch.cat(self._traj, dim=1)[:, :-1, :]
            self._traj = torch.cat(self._traj, dim=1)
            print(f"Shape after concatenation: {self._traj.shape}")
        return self._traj    

    @property
    def fwd_probs(self):
        if isinstance(self._fwd_probs, list):
            for i, tensor in enumerate(self._fwd_probs):
                print(f"Shape of tensor {i}: {tensor}")
            print(f"Fwd Probs Shape before cat: {len(self._fwd_probs)})")
            self._fwd_probs = torch.stack(self._fwd_probs, dim=0)
            self._fwd_probs = self._fwd_probs.t()
            print(f"Fwd Probs Shape before traj balance: {self._fwd_probs.shape}")
        return self._fwd_probs
    
    @property
    def actions(self):
        if isinstance(self._actions, list):
            for i, tensor in enumerate(self._actions):
                print(f"Shape of tensor {i}: {tensor}")

            self._actions = torch.stack(self._actions, dim=0)
        return self._actions
    
    @property
    def back_probs(self):
        if self._back_probs is not None:
            return self._back_probs
        
        print(f"Shape of Trajectory at Start of back_probs method {self.traj.shape}")
        #s = self.traj[:, 1:, :].reshape(-1, self.env.state_dim)
        s = self.traj[:, 1:, :]
        print(f"Shape of S--Trajectory back_probs reshape: {s.shape}")
        #prev_s = self.traj[:, :-1, :].reshape(-1, self.env.state_dim)
        prev_s = self.traj[:, :-1, :]
        print(f"Shape of Previous S--Trajectory back_probs reshape: {prev_s.shape}")
        #traj_manipulator = SparseTensorManipulator(self.traj, self.env.state_dim)
        #s, prev_s = traj_manipulator.get_s_and_prev_s()
        print(f"actions shape {self.actions.shape}")
        #actions = self.actions[:, :-1].flatten()
        actions = self.actions.t()
        #actions = actions[:, :-1]
        
        terminated = (actions == -1) | (actions == self.env.num_actions - 1)
        print(f"Terminated in Back Probs shape: {terminated.shape}")
        zero_to_n = torch.arange(len(terminated))
        #actions_to_n = torch.arange(len(actions[1]))
        print(f"Shape of S in back_probs: {s.shape}")
        #print(f"Zero to n shape: {zero_to_n.shape}, {zero_to_n}")
        #back_probs = self.backward_policy(s, self.env) * self.env.mask(prev_s)
        back_probs = self.backward_policy(s, self.env, terminated)
        print(f"Env Prevs Mask Shape {self.env.mask(prev_s).shape}")
        print(f"Back Probs Shape before where: {back_probs.shape}")        
        #print(f"back probs subset shape before where: {back_probs[zero_to_n].shape}")
        #This is not what we want. It should be a tensor that has dim 0 of batch size and dim 1 of probs. 
        #back_probs = torch.where(terminated, 1, back_probs[zero_to_n, actions])
        #back_probs = torch.where(terminated, 1, back_probs)
        print(f"back probs shape after where: {back_probs.shape}")
        self._back_probs = back_probs.reshape(self.num_samples, -1)
        
        return self._back_probs