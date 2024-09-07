import torch
from torch import Tensor
from typing import Tuple, List
import numpy as np
from torch.nn.functional import one_hot
from torch_geometric.data import Data
from gflownet.env import Env
from gflownet.utils import update_edges_and_convert_to_sparse, resize_sparse_tensor
import gc

class PreconditionerEnv(Env):
    def __init__(self, matrix_size: int, initial_matrix: Tensor, original_matrix: Tensor):
        self.matrix_size = matrix_size
        self.init_nnz = initial_matrix.coalesce().indices().size(1)
        self.state_dim = self.init_nnz # Modify for current use case
        self.num_actions = (self.init_nnz) + 1  # Number of actions is the number of entries in the matrix plus one for terminal action.
        self.matrix = initial_matrix.clone()
        self.original_matrix = original_matrix.clone()
        #self.init_mask = self.create_mask_from_sparse_matrix(self.matrix)


        # Extract edge_index and edge_attr directly from the sparse matrix
        edge_index = self.matrix._indices()
        edge_attr = self.matrix._values()
        self.data = Data(edge_index=edge_index, edge_attr=edge_attr.float())

        #Calculate initial metrics
        self.orig_residual = self.calculate_residual(self.original_matrix, self.original_matrix)
        self.orig_flops, _ = self.matrix_flops(self.original_matrix)


    def update(self, sparse_matrices: List[torch.sparse.FloatTensor], actions: List[List[int]], alpha: float) -> List[torch.sparse.FloatTensor]:
        batch_size = len(actions)
        #print(f"Action size: {type(actions)}")
        rewards = []

        for i in range(batch_size):
            good_actions = []
            [good_actions.append(x) for x in actions[i] if x != -1]
            num_good_actions = int(len(good_actions))
            #print(f"Non-terminating action size for sample {i}: {len([x for x in actions[i] if x != -1])}")

            updated_matrix = update_edges_and_convert_to_sparse(self.data, good_actions, self.matrix_size)

            resized_matrix = resize_sparse_tensor(updated_matrix, (self.matrix_size, self.matrix_size))
            
            reward = self.reward(resized_matrix, num_good_actions, alpha)
            rewards.append(reward)
            del updated_matrix
            del resized_matrix
            gc.collect()
        return rewards 


    def reward(self, s: Tensor, traj_length: int, alpha: float) -> float:
        # Use the current matrix as a preconditioner and calculate the reward
        # based on its performance (e.g., reduced iterations, improved stability)
        #Need to figure out a way to penalize a trajectory to avoid the model picking a blank matrix. For now, dividing by log length of trajectory, but will need to come up with something.
        reward = self.evaluate_preconditioner(s, self.original_matrix, self.orig_residual, self.orig_flops, alpha)
        #print(f"Reward Method before Traj Length: {reward}")

        #reward = reward/torch.log(torch.tensor(traj_length, dtype=torch.float64))
        #reward = (reward/torch.tensor(traj_length, dtype=torch.float64)) * 100
        reward = reward.to(torch.float64) * 1000
        #print(f"Reward Method after Traj Length: {reward}")
        return reward
    
    def matrix_flops(self, matrix: Tensor) -> Tuple[int, int]:
        if matrix.is_sparse:
            # For sparse tensors, the number of non-zero elements is the size of the values tensor
            non_zeros = matrix._values().numel()
            flops = non_zeros * matrix.shape[1] * 2
        else:
            # We should be working exclusively with sparse tensors, so maybe replace this if/else statement with an assert. 
            non_zeros = torch.nonzero(matrix).size(0)
            flops = 2 * non_zeros        
        return flops, non_zeros

    def calculate_residual(self, updated_matrix: Tensor, original_matrix: Tensor) -> Tensor:
        #Calculate residual term, moving to its own method because we only want to calculate ||A*A-I|| once
        # Residual term: ||M*A - I||
        i = torch.arange(0, self.matrix_size)
        i = torch.stack([i, i])
        v = torch.ones(self.matrix_size, dtype=torch.float64)
        sparse_identity = torch.sparse_coo_tensor(i, v, (self.matrix_size, self.matrix_size))
        #print(f"updated_matrix_dtype: {updated_matrix.dtype}")
        #print(f"original_matrix_dtype: {original_matrix.dtype}")
        product = torch.mm(updated_matrix, original_matrix)
        #print(f"Product {product}")
        residual = torch.norm(product - sparse_identity)
        #print(f"residual {residual}")

        return residual

    def mask(self, s: Tensor) -> Tensor:
        # Implement your masking logic here
        # This is an example implementation that allows all actions. We can implement this later if it is needed.
        return torch.ones(len(s), self.num_actions)

    
    def create_mask_from_sparse_matrix(self, sparse_matrix):
        """
        Create a mask that will keep non-zero values and zero out indices that map to zero values in the sparse matrix.
        
        Args:
            sparse_matrix (torch.sparse.FloatTensor): The sparse matrix.
            
        Returns:
            torch.Tensor: A mask of the same size as the sparse matrix.
        """
        if not sparse_matrix.is_sparse:
            raise ValueError("The input tensor must be a sparse tensor.")
        
        # Extract the size of the sparse matrix
        size = sparse_matrix.size()
        
        # Initialize a mask with zeros
        mask = torch.zeros(size, dtype=torch.float32)
        
        # Get the indices of non-zero elements in the sparse matrix
        indices = sparse_matrix._indices()
        
        # Set the corresponding entries in the mask to one
        mask[indices[0], indices[1]] = 1

        #flat_mask = mask.view(-1)

        #resized_mask = flat_mask.view((1, size[0] * size[1]))
        resized_mask = mask.view(1, size[0] * size[1])

        additional_entry = torch.tensor([[1]], dtype=torch.float32)

        extended_mask = torch.cat((resized_mask, additional_entry), dim=1)
        #print(f"extended_mask {extended_mask}")
        return extended_mask
    
    def evaluate_preconditioner(self, updated_matrix: Tensor, original_matrix: Tensor, orig_residual: float, orig_flops: int, alpha: float) -> float:

        
        # Compute the computational cost (floating point operations - FLOPs)
        # For simplicity, we can assume 2 FLOPs for each non-zero element (one for multiplication and one for addition)
        # in the original matrix when multiplied by the preconditioner
        # Assuming 'matrix' is the sparse matrix
        
        residual = self.calculate_residual(updated_matrix, original_matrix)
        #print(f"Updated Matrix Flops Shape: {updated_matrix.shape}")
        #print(f"Updated Matrix NNZ {updated_matrix._nnz()}")
        flops, non_zeros = self.matrix_flops(updated_matrix)
        
        # Performance metric
        #inverse_residual = 1 / torch.log((1 + residual))
        #Modified so that residual is the denominator as it is expected to grow. We may need to change this.
        #residual_ratio = orig_residual / residual if residual != 0 else float('inf')
        residual_ratio = residual / orig_residual if orig_residual != 0 else float('inf')
        #print(f"Residual for Updated Matrix {residual}")
        #print(f"Original Residual: {orig_residual}")
        #print(f"residual_ratio {residual_ratio}")
        computational_ratio = flops / orig_flops if orig_flops != 0 else float('inf')
        #print(f"No. flops for Updated Matrix {flops}")
        #print(f"No. Flops Original Matrix {orig_flops}")
        #print(f"computational_ratio: {computational_ratio}")
        
        performance_metric = self.alpha * (1 - residual_ratio) + (1 - self.alpha) * (1 - computational_ratio)
        #print(f"performance metric {performance_metric}")
        return performance_metric
