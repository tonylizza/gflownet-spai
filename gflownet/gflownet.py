import torch
from torch import nn
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.distributions import Categorical
import torch.nn.utils.rnn as rnn_utils
from torch_geometric.data import Data
import pytorch_lightning as pl
from .log import Log
from .utils import resize_sparse_tensor, log_memory_usage, malloc_usage, calculate_reward, update_edges_and_convert_to_sparse, trajectory_balance_loss, decompose_ilu_and_create_linear_operator, custom_solve_with_modified_LU
from typing import List
from .dataset import *
import gc
from tqdm import tqdm
from .validate import solve_with_gmres
from .utils import torch_sparse_to_csr, convert_sparse_idx_to_row_col
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import gmres, LinearOperator, spilu
import pandas as pd
from datetime import datetime
import csv

class GFlowNet(pl.LightningModule):
    def __init__(self, forward_policy, backward_policy, no_sampling_batch = 32, lr = .00002, schedule_patience=5):
        super().__init__()
        #self.total_flow = Parameter(torch.ones(1))
        self.automatic_optimization = False
        self.register_buffer('total_flow', torch.ones(1))
        self.forward_policy = forward_policy
        self.backward_policy = backward_policy
        self.no_sampling_batch = no_sampling_batch
        self.padding_value = 1e-9
        self.lr = lr
        self.schedule_patience = schedule_patience
    
    def forward(self, batch):
        trajectories = []
        rewards = []
        forward_probs = []
        fwd_state_flows = []
        #backward_probs = []
        selected_fwd_actions_list = []
        alphas = []
        iter = 1
        for sample in range(self.no_sampling_batch):
            # 1. Generate the trajectory for the current sample
            #print(f"Iter: {iter}")
            iter = iter + 1
            trajectory = []
            fwd_state_flow = []
            action_probs_list = []
            selected_fwd_actions = []
            current_state = batch['data']
            #print(f"Num Actions: {current_state.edge_attr.size(0) + 1}")
            #print(f"Current State Size {current_state.edge_attr.size(0)}")
            terminated = False
            sampling = 0
            max_samples = min(8000, current_state.edge_attr.size(0) // 2)
            while not terminated:
                #print(f"Sampling: {sampling}")
                sampling = sampling + 1
                action_probs, flow, alpha = self.forward_policy(current_state, actions=trajectory)
                #print(f"Flow Grad: {flow.requires_grad}")
                #print(f"Action Probs {action_probs.shape}")
                action_distribution = torch.distributions.Categorical(action_probs)
                action = action_distribution.sample()
                selected_action_prob = action_probs[:, action]
                selected_fwd_actions.append(selected_action_prob)
                #selected_action_prob.detach()
                #print(f"Action {action}")
                trajectory.append(action)
                fwd_state_flow.append(flow)
                alphas.append(alpha)
                action_probs_list.append(action_probs)
                #print(action_probs)
                # Append action to the trajectory

                # Check if the action is the termination action
                if action == (action_probs.shape[-1] - 1) or sampling >= max_samples:
                    #print(f"action value: {action}")
                    terminated = True
                if sampling % 1000 == 0:
                    log_memory_usage(f"Sampled {sampling} actions")
                    print(len(action_probs_list))

            # 2. Use the trajectory to update the matrix and calculate the reward
            alpha = torch.stack(alphas).mean(dim=0)
            with torch.no_grad():
                reward = self.update_and_compute_reward(batch['data'], trajectory, batch['starting_matrix'], batch['starting_residual'], batch['starting_flops'], batch['matrix_sq_side'], alpha)
            trajectories.append(torch.tensor(trajectory))
            selected_fwd_actions_list.append(torch.stack(selected_fwd_actions))
            fwd_state_flows.append(torch.stack(fwd_state_flow, dim=0))
            print(reward)
            rewards.append(reward)

            # 3. Optionally, calculate the forward/backward probabilities (for trajectory balance loss)
            forward_probs.append(torch.stack(action_probs_list))

        padded_trajectories = rnn_utils.pad_sequence(trajectories, batch_first=True, padding_value=-1)
        selected_fwd_actions_list = [torch.tensor(action) if not isinstance(action, torch.Tensor) else action for action in selected_fwd_actions_list]
        padded_forward_flows = rnn_utils.pad_sequence(fwd_state_flows, batch_first=True, padding_value=0.0)
        #print(f"Padded Forward Flows Grad: {padded_forward_flows.requires_grad}")
        padded_selected_fwds = rnn_utils.pad_sequence(selected_fwd_actions_list, batch_first=True, padding_value=0.0).squeeze(-1)    
        #backward_probs.append(self.backward_policy(padded_trajectories))  # Backward policy can depend on the trajectory

        valid_actions_list = []
        termination_action_index = batch['data'].edge_attr.size(0) + 1
        for i in range(self.no_sampling_batch):
            sample_trajectory = padded_trajectories[i]
            # Exclude -1 and get unique actions
            valid_actions = sample_trajectory[sample_trajectory != -1].unique().tolist()
            # Include termination action
            #valid_actions.append(termination_action_index)
            valid_actions_list.append(valid_actions)
        backward_probs, backward_state_flows = self.backward_policy(trajectories=padded_trajectories, valid_actions_list=valid_actions_list, termination_action_index=termination_action_index)
        #print(f"Backward Probs Orig {backward_probs.shape}")
        # Use Categorical to sample actions from the backward probabilities
        # backward_probs shape: [number_batch, max_trajectory_length, num_actions]
        distribution = torch.distributions.Categorical(probs=backward_probs.squeeze(-2))
        selected_actions = distribution.sample().unsqueeze(-1)
        # backward_prob shape: [number_batch, max_trajectory_length, 1, num_actions]
        # selected_actions shape: [number_batch, max_trajectory_length, 1] (actual actions taken)

        # Ensure selected_actions has the right shape for indexing
        actions_expanded = selected_actions.unsqueeze(-1)  # Shape becomes [number_batch, max_trajectory_length, 1, 1]

        # Use torch.gather to index into backward_prob and extract probabilities of the selected actions
        selected_back_probs = torch.gather(backward_probs, dim=-1, index=actions_expanded).squeeze(-1)

        # selected_back_probs will now be of shape [number_batch, max_trajectory_length, 1] containing the probabilities for the selected actions

        #print(f"Backward Probs Resize {backward_probs.shape}")
        padded_forward_probs = rnn_utils.pad_sequence(forward_probs, batch_first=True, padding_value=0)
        #padded_forward_probs.detach()
        print(f"Padded Forward Probs {padded_forward_probs.shape}")



        return {
            "trajectories": trajectories,
            "rewards": torch.tensor(rewards),
            "forward_probs": padded_forward_probs,
            "backward_probs": backward_probs,
            "padded_forward_flows": padded_forward_flows,
            "padded_selected_fwds": padded_selected_fwds,
            "selected_back_probs": selected_back_probs
        }


    def training_step(self, batch, batch_idx):
        # Call the forward method to generate trajectories, rewards, forward_probs, and backward_probs
        output = self.forward(batch)

        # Extract the necessary components from the forward pass
        trajectories = output["trajectories"]
        rewards = output["rewards"]
        forward_probs = output["forward_probs"]
        backward_probs = output["backward_probs"]
        selected_fwd_probs = output['padded_selected_fwds']
        selected_back_probs = output['selected_back_probs']
        #print(f"Selected Fwd Probs Grad {selected_fwd_probs.shape}")
        #print(f"Selected Back Probs Grad {selected_back_probs.shape}")
        #print(f"Rewards {rewards}")
        padded_forward_flows = output["padded_forward_flows"]
        #print(f"Forward Probs Shape {forward_probs}")
        #print(f"Back Probs Shape {backward_probs}")
        avg_reward = rewards.mean()

        # Compute the loss based on trajectory balance (or another loss function)
        loss = trajectory_balance_loss(rewards, selected_fwd_probs, selected_back_probs)
        print(f"Loss: {loss}")

        reward_per_loss = avg_reward/loss
        self.log('avg_reward', avg_reward, on_epoch=True)
        # Log the loss and return it
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('reward_per_loss', reward_per_loss, on_epoch=True)

            # Perform backward pass
        self.manual_backward(loss)
    
        #self.check_gradients()

        # Log the action distribution (histogram of probabilities)
        for i, probs in enumerate(forward_probs):
            self.logger.experiment.add_histogram(f'action_distribution_{i}', probs, self.current_epoch)

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
            patience=self.schedule_patience,      # How many epochs to wait before reducing LR
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

    def check_gradients(self):
        print("\nChecking Gradients and Parameters")
        for name, param in self.named_parameters():
            if param.grad is not None:
                print(f"Parameter: {name}")
                print(f"Values: {param}")
                print(f"Gradients: {param.grad}")
            else:
                print(f"Parameter: {name} has no gradients.")

    def on_before_backward(self, loss):
        # Collect gradients only for parameters that have non-None gradients
        #self.check_gradients()
        grads = [p.grad.detach() for p in self.parameters() if p.grad is not None]
        
        if len(grads) > 0:
            # Compute the gradient norm if there are valid gradients
            grad_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in grads]))
            self.log('grad_norm', grad_norm)
        else:
            # Optionally, log a warning or handle the case where no gradients are available
            self.log('grad_norm', torch.tensor(0.0))

    def on_train_end(self):
        """
        This method is called after training is complete.
        We use it to run validation after all training epochs are done.
        """
        results = self.run_validation()
        aggregated_results = self.aggregate_validation_results(results[0])
        # Manually log the results using logger
        if self.logger:
            # Assuming you're using TensorBoard, WandB, etc.
            self.logger.experiment.add_scalar('avg_original_iterations', aggregated_results['avg_original_iters'], global_step=self.current_epoch)
            self.logger.experiment.add_scalar('avg_original_time', aggregated_results['avg_original_time'], global_step=self.current_epoch)
            self.logger.experiment.add_scalar('avg_ilu_iterations', aggregated_results['avg_ilu_iters'], global_step=self.current_epoch)
            self.logger.experiment.add_scalar('avg_ilu_time', aggregated_results['avg_ilu_time'], global_step=self.current_epoch)
            self.logger.experiment.add_scalar('avg_sparse_iterations', aggregated_results['avg_sparse_iters'], global_step=self.current_epoch)
            self.logger.experiment.add_scalar('avg_sparse_time', aggregated_results['avg_sparse_time'], global_step=self.current_epoch)

        # If you are using other loggers (like WandB), you may need to adapt this accordingly
        self.save_validation_results_to_csv(results[0])
        print(f"Logged validation results after training epoch {self.current_epoch}")

    def run_validation(self):
        """
        Manually run validation using the validation data loader.
        """
        self.forward_policy.eval()  # Set the model to evaluation mode

        # Loop over the validation dataloader
        results = []
        with torch.no_grad():
            for batch in tqdm(self.trainer.datamodule.val_dataloader()):
                
                _, _, original_iterations, original_time = solve_with_gmres(batch['original_csr'], batch['b_vector'], max_iters=batch['matrix_sq_side'])
                print(f"GMRES no preconditioner")
                ilu = spilu(batch['original_csr'], permc_spec = "NATURAL")
                L = ilu.L
                U = ilu.U
                
                M_x = lambda x: ilu.solve(x)
                M = LinearOperator(batch['original_csr'].shape, lambda x: custom_solve_with_modified_LU(x, L, U))
                _, _, ilu_iterations, ilu_time = solve_with_gmres(batch['original_csr'], batch['b_vector'], M=M, max_iters=batch['matrix_sq_side'])
                print(f"GMRES orig preconditioner")
                batch_results = []
                for _ in range(10):
                    sampled_trajectory = self.sample_trajectory(batch)
                    L_copy = ilu.L.copy().tocoo()
                    U_copy = ilu.U.copy().tocoo()
                    for i in sampled_trajectory:
                        row = batch['data'].edge_index[0][i]
                        col = batch['data'].edge_index[1][i]
                        #row, col = convert_sparse_idx_to_row_col(i, batch['matrix_sq_side'])
                        if row != col:
                            L_copy.data[(L_copy.row == row) & (L_copy.col == col)] = 0
                            U_copy.data[(U_copy.row == row) & (U_copy.col == col)] = 0
                    #sampled_matrix = update_edges_and_convert_to_sparse(batch['data'], sampled_trajectory, batch['matrix_sq_side'])
                    #sampled_csr = torch_sparse_to_csr(sampled_matrix, batch['matrix_sq_side'])
                    L_copy = L_copy.tocsc()
                    U_copy = U_copy.tocsc()
                    #sampled_M_x = lambda x: ilu.solve(x)
                    sampled_M = LinearOperator(batch['original_csr'].shape, lambda x: custom_solve_with_modified_LU(x, L_copy, U_copy))
                    #sampled_M = decompose_ilu_and_create_linear_operator(sampled_csr)
                    _, _, sparse_iterations, sparse_time = solve_with_gmres(batch['original_csr'], batch['b_vector'], M=sampled_M, max_iters=batch['matrix_sq_side'])
                    print(f"GMRES sparse preconditioner")
                    filename = batch['filename']
                    num_non_zeros = batch['original_csr'].nnz
                    trajectory_length = len(sampled_trajectory)

                    
                    batch_results.append({
                        "filename": filename,
                        "num_non_zeros": num_non_zeros,
                        "trajectory_length": trajectory_length,
                        "original_iterations": original_iterations,
                        "original_time": original_time,
                        "ilu_iterations": ilu_iterations,
                        "ilu_time": ilu_time,
                        "sparse_iterations": sparse_iterations,
                        "sparse_time": sparse_time,
                    })

        results.append(batch_results)
        return results
        # After validation, aggregate and log the results
        

    def aggregate_validation_results(self, outputs):
        """
        Aggregate the results from validation and log them.
        """
        original_iters = [x['original_iterations'] for x in outputs]
        original_times = [x['original_time'] for x in outputs]
        ilu_iters = [x['ilu_iterations'] for x in outputs]
        ilu_times = [x['ilu_time'] for x in outputs]
        sparse_iters = [x['sparse_iterations'] for x in outputs]
        sparse_times = [x['sparse_time'] for x in outputs]

        # Convert to floating point before computing mean
        avg_original_iters = torch.mean(torch.tensor(original_iters).float())
        avg_original_time = torch.mean(torch.tensor(original_times).float())
        avg_ilu_iters = torch.mean(torch.tensor(ilu_iters).float())
        avg_ilu_time = torch.mean(torch.tensor(ilu_times).float())                          
        avg_sparse_iters = torch.mean(torch.tensor(sparse_iters).float())
        avg_sparse_time = torch.mean(torch.tensor(sparse_times).float())

        return {
            'avg_original_iters': avg_original_iters,
            'avg_original_time': avg_original_time,
            'avg_ilu_iters': avg_ilu_iters,
            'avg_ilu_time': avg_ilu_time,
            'avg_sparse_iters': avg_sparse_iters,
            'avg_sparse_time': avg_sparse_time
        }


    def sample_trajectory(self, batch):
        """
        Function to sample a trajectory from the GFlowNet model.
        """
        trajectory = []
        current_state = batch['data']
        terminated = False
        while not terminated:
            action_probs, flow, alpha = self.forward_policy(current_state, actions=trajectory)
            action_distribution = Categorical(action_probs)
            action = action_distribution.sample()
            trajectory.append(action)

            if action == (action_probs.shape[-1] - 1):
                terminated = True
        termination_action_index = batch['data'].edge_attr.size(0)
        trajectory = [int(action.item()) for action in trajectory if action != termination_action_index]
        return trajectory

    def save_validation_results_to_csv(self, validation_results):
        # Get current datetime for filename
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        csv_filename = f'validation_results_{current_time}.csv'

        # Define the header for the CSV
        csv_header = [
            'filename', 'num_non_zeros', 'trajectory_length',
            'original_iterations', 'original_time',
            'ilu_iterations', 'ilu_time',
            'sparse_iterations', 'sparse_time'
        ]

        # Open the CSV file in write mode
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=csv_header)
            
            # Write the header
            writer.writeheader()
            
            # Write the validation results to the CSV file
            for result in validation_results:
                writer.writerow(result)

        print(f"Validation results saved to {csv_filename}")