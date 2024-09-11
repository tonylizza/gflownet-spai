import torch
from torch import nn
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.distributions import Categorical
import torch.nn.utils.rnn as rnn_utils
from torch_geometric.data import Data
import pytorch_lightning as pl
from .log import Log
from .utils import resize_sparse_tensor, log_memory_usage, malloc_usage, calculate_reward, update_edges_and_convert_to_sparse, trajectory_balance_loss
from typing import List
from .dataset import *
import gc

class GFlowNet(pl.LightningModule):
    def __init__(self, forward_policy, backward_policy, no_sampling_batch = 32, lr = .00002):
        super().__init__()
        #self.total_flow = Parameter(torch.ones(1))
        self.register_buffer('total_flow', torch.ones(1))
        self.forward_policy = forward_policy
        self.backward_policy = backward_policy
        self.no_sampling_batch = no_sampling_batch
        self.padding_value = 1e-9
        self.lr = lr
    
    def forward(self, batch):
        trajectories = []
        rewards = []
        forward_probs = []
        #backward_probs = []
        alphas = []
        iter = 1
        for sample in range(self.no_sampling_batch):
            # 1. Generate the trajectory for the current sample
            #print(f"Iter: {iter}")
            iter = iter + 1
            trajectory = []
            action_probs_list = []
            current_state = batch['data']
            terminated = False
            sampling = 1
            while not terminated:
                #print(f"Sampling: {sampling}")
                sampling = sampling + 1
                action_probs, alpha = self.forward_policy(current_state, actions=trajectory)
                #print(f"Action Probs {action_probs}")
                action_distribution = torch.distributions.Categorical(action_probs)
                action = action_distribution.sample()
                #print(f"Action {action}")
                trajectory.append(action)
                alphas.append(alpha)
                action_probs_list.append(action_probs)
                # Append action to the trajectory

                # Check if the action is the termination action
                if action == (action_probs.shape[-1] - 1):
                    terminated = True


            # 2. Use the trajectory to update the matrix and calculate the reward
            alpha = torch.stack(alphas).mean(dim=0)
            reward = self.update_and_compute_reward(batch['data'], trajectory, batch['starting_matrix'], batch['starting_residual'], batch['starting_flops'], batch['matrix_sq_side'], alpha)
            trajectories.append(torch.tensor(trajectory))
            rewards.append(reward)

            # 3. Optionally, calculate the forward/backward probabilities (for trajectory balance loss)
            forward_probs.append(torch.stack(action_probs_list))

        padded_trajectories = rnn_utils.pad_sequence(trajectories, batch_first=True, padding_value=-1)    
        #backward_probs.append(self.backward_policy(padded_trajectories))  # Backward policy can depend on the trajectory

        backward_probs = self.backward_policy(padded_trajectories)
        backward_probs = backward_probs.reshape(self.no_sampling_batch, -1)
        padded_forward_probs = rnn_utils.pad_sequence(forward_probs, batch_first=True, padding_value=1e-9)



        return {
            "trajectories": trajectories,
            "rewards": torch.tensor(rewards),
            "forward_probs": padded_forward_probs,
            "backward_probs": backward_probs
        }


    def training_step(self, batch, batch_idx):
        # Call the forward method to generate trajectories, rewards, forward_probs, and backward_probs
        output = self.forward(batch)

        # Extract the necessary components from the forward pass
        trajectories = output["trajectories"]
        rewards = output["rewards"]
        forward_probs = output["forward_probs"]
        backward_probs = output["backward_probs"]

        print(f"Forward Probs Shape {forward_probs.shape}")
        print(f"Back Probs Shape {backward_probs.shape}")

        # Compute the loss based on trajectory balance (or another loss function)
        loss = trajectory_balance_loss(self.total_flow, rewards, forward_probs, backward_probs)
        print(f"Loss: {loss}")

        # Log the loss and return it
        self.log('train_loss', loss, on_step=True)
        return loss


    
    def update_and_compute_reward(self, data: Data, actions: List[int], starting_matrix, orig_residual: int, orig_flops: int, matrix_size: int, alpha: float) -> torch.Tensor:
        batch_size = len(actions)
        rewards = []

        updated_matrix = update_edges_and_convert_to_sparse(data, actions, matrix_size)

        resized_matrix = resize_sparse_tensor(updated_matrix, (matrix_size, matrix_size))
            
        reward = calculate_reward(starting_matrix, resized_matrix, orig_residual, orig_flops, len(actions), alpha)
        del updated_matrix
        del resized_matrix
        gc.collect()
        return reward
  
    def configure_optimizers(self):
        # Set up your optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        # Set up your learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',      # We want to minimize the loss
            factor=0.2,      # How much to reduce the learning rate by
            patience=5,      # How many epochs to wait before reducing LR
            verbose=True,    # Print a message when LR is reduced
        )
        
        # Return both optimizer and scheduler
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss'  # The metric to monitor for reducing LR
            }
        }
    

