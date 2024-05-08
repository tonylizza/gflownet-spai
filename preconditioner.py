import torch
import numpy as np
from torch.nn.functional import one_hot
from gflownet.env import Env

class PreconditionerEnv(Env):
    def __init__(self, matrix_size, initial_matrix):
        self.matrix_size = matrix_size
        self.state_dim = matrix_size**2
        self.num_actions = matrix_size**2  # considering each entry as a potential action
        self.matrix = initial_matrix
        
    def update(self, s, actions):
        # Convert action index to matrix row and column
        row, col = divmod(actions.item(), self.matrix_size)
        # Perform the action: remove or modify the entry in the matrix
        self.matrix[row, col] = 0  # example action: remove the entry
        return self.matrix.flatten().float()
    
    def mask(self, s):
        # Mask actions that would be invalid (e.g., removing already removed entries)
        mask = self.matrix.flatten() > 0
        return mask.float()
        
    def reward(self, s):
        # Use the current matrix as a preconditioner and calculate the reward
        # based on its performance (e.g., reduced iterations, improved stability)
        reward = self.evaluate_preconditioner(self.matrix)
        return reward
    
    def matrix_flops(self, matrix):
        if matrix.is_sparse:
            # For sparse tensors, the number of non-zero elements is the size of the values tensor
            non_zeros = matrix._values().numel()
            flops = non_zeros * matrix.shape[1] * 2
        else:
            # For dense tensors, use torch.nonzero or equivalent to count non-zeros
            #non_zeros = torch.nonzero(matrix, as_tuple=False).size(0)
            non_zeros = torch.nonzero(matrix).size(0)
            flops = 2 * non_zeros        
        return flops, non_zeros

    def calculate_residual(self, updated_matrix, original_matrix):
        #Calculate residual term, moving to its own method because we only want to calculate ||A*A-I|| once
        # Residual term: ||M*A - I||
        i = torch.arange(0, self.matrix_size)
        i = torch.stack([i, i])
        v = torch.ones(self.matrix_size, dtype=torch.float64)
        sparse_identity = torch.sparse_coo_tensor(i, v, (self.matrix_size, self.matrix_size))
        residual = torch.norm(torch.mm(updated_matrix, original_matrix) - sparse_identity)

        return residual

    def evaluate_preconditioner(self, updated_matrix, original_matrix, orig_residual, orig_flops, lambda_hyperparam, alpha):

        
        # Compute the computational cost (floating point operations - FLOPs)
        # For simplicity, we can assume 2 FLOPs for each non-zero element (one for multiplication and one for addition)
        # in the original matrix when multiplied by the preconditioner
        # Assuming 'matrix' is the sparse matrix

        residual = self.calculate_residual(updated_matrix, original_matrix)

        flops, non_zeros = self.matrix_flops(updated_matrix)

        # Compute the penalty for non-zeros in the preconditioner
        penalty = lambda_hyperparam * non_zeros
        
        # Performance metric
        #inverse_residual = 1 / torch.log((1 + residual))
        residual_ratio = residual / orig_residual if orig_residual != 0 else float('inf')
        computational_ratio = flops / orig_flops if orig_flops != 0 else float('inf')
        
        performance_metric = alpha * (1-residual_ratio) + (1 - alpha) *(1 - computational_ratio)
        
        #return performance_metric
        return (residual_ratio, computational_ratio, performance_metric)
