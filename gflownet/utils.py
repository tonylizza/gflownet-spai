import torch
from torch import Tensor
from torch_geometric.data import Data
from typing import Tuple, List
import scipy.io
import psutil
import tracemalloc

class SparseTensorManipulator:
    def __init__(self, sparse_tensor, state_dim):
        self.sparse_tensor = sparse_tensor.coalesce()  # Ensure the tensor is coalesced
        self.state_dim = state_dim

    def slice_and_reshape_sparse(self, start, end):
        indices = self.sparse_tensor.indices()
        values = self.sparse_tensor.values()

        # Mask for selecting indices within the specified range
        mask = (indices[1] >= start) & (indices[1] < end)
        new_indices = indices[:, mask]
        new_values = values[mask]

        # Adjust the indices after slicing
        new_indices[1] -= start

        #print(f"End start: {end, start}")
        #print(f"New indices shape {new_indices.shape}")
        #print(f"New values shape {new_values}")

        # Create the new sparse tensor with adjusted indices and values
        new_sparse_tensor = torch.sparse_coo_tensor(new_indices, new_values, (self.sparse_tensor.size(0), end - start, self.sparse_tensor.size(2)))

        return new_sparse_tensor

    def get_s_and_prev_s(self):
        # Slice and reshape the sparse tensor to get s and prev_s
        s = self.slice_and_reshape_sparse(1, self.sparse_tensor.size(1))
        prev_s = self.slice_and_reshape_sparse(0, self.sparse_tensor.size(1) - 1)

        # Reshape the sparse tensors to the required dimensions
        s_reshaped = s.to_dense().reshape(-1, self.state_dim).to_sparse()
        prev_s_reshaped = prev_s.to_dense().reshape(-1, self.state_dim).to_sparse()

        return s_reshaped, prev_s_reshaped

def matrix_to_graph(matrix: Tensor) -> Data:
    # Utility function that converts a PyTorch sparse tensor to a PyTorch Geometric graph
    edge_index = torch.stack([matrix._indices()[0], matrix._indices()[1]], dim=0)
    edge_attr = matrix._values()
    num_nodes = matrix.size(0)
    x = torch.ones((num_nodes, 1))  # Example node features
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def market_matrix_to_sparse_tensor(file_path: str) -> Tensor:
    # Load the Matrix Market file into a scipy sparse matrix
    matrix = scipy.io.mmread(file_path).tocoo()
    
    # Convert the scipy sparse matrix to PyTorch sparse tensor
    values = torch.tensor(matrix.data, dtype=torch.float64)
    indices = torch.tensor([matrix.row, matrix.col], dtype=torch.long)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, matrix.shape)
    
    return sparse_tensor
'''
def resize_sparse_tensor(sparse_tensor: torch.Tensor, new_size: tuple) -> torch.Tensor:
    """
    Resizes a sparse tensor to the new size without converting it to a dense tensor.
    
    Args:
        sparse_tensor (torch.Tensor): The original sparse tensor.
        new_size (tuple): The desired size of the new sparse tensor.
    
    Returns:
        torch.Tensor: The new sparse tensor with the specified size.
    """
    if not sparse_tensor.is_sparse:
        raise ValueError("The input tensor must be a sparse tensor.")
    
    # Extract the indices and values
    indices = sparse_tensor._indices()
    values = sparse_tensor._values()
    
    # Create a new sparse tensor with the same non-zero elements but a different size
    new_sparse_tensor = torch.sparse_coo_tensor(indices, values, new_size, dtype=torch.double)
    
    return new_sparse_tensor
'''

def resize_sparse_tensor(sparse_tensor: torch.Tensor, new_size: tuple) -> torch.Tensor:
    """
    Resizes a sparse tensor to the new size without converting it to a dense tensor.
    
    Args:
        sparse_tensor (torch.Tensor): The original sparse tensor.
        new_size (tuple): The desired size of the new sparse tensor.
    
    Returns:
        torch.Tensor: The new sparse tensor with the specified size.
    """
    if not sparse_tensor.is_sparse:
        raise ValueError("The input tensor must be a sparse tensor.")
    
    # Extract the indices and values
    indices = sparse_tensor._indices()
    values = sparse_tensor._values()
    
    old_size = sparse_tensor.size()
    old_num_elements = old_size[-1]
    new_num_rows, new_num_cols = new_size
    
    if old_num_elements != new_num_rows * new_num_cols:
        raise ValueError("The total number of elements must match between the old and new sizes.")
    
    # Flatten the indices for the old 1D shape to 2D for the new shape
    if len(old_size) == 2 and old_size[0] == 1:
        linear_indices = indices[1]  # Only the second row has relevant indices in the [1, 324] shape
        row_indices = linear_indices // new_num_cols
        col_indices = linear_indices % new_num_cols
        new_indices = torch.stack([row_indices, col_indices], dim=0)
    else:
        raise ValueError("Unsupported resize operation.")
    
    # Create a new sparse tensor with the same non-zero elements but a different size
    new_sparse_tensor = torch.sparse_coo_tensor(new_indices, values, new_size, dtype=sparse_tensor.dtype).coalesce()
    
    return new_sparse_tensor
    
def resize_sparse_tensor_to_flat(sparse_tensor: torch.Tensor, new_size: tuple) -> torch.Tensor:
    """
    Resizes a sparse tensor to the new size without converting it to a dense tensor.
    
    Args:
        sparse_tensor (torch.Tensor): The original sparse tensor.
        new_size (tuple): The desired size of the new sparse tensor.
    
    Returns:
        torch.Tensor: The new sparse tensor with the specified size.
    """
    if not sparse_tensor.is_sparse:
        raise ValueError("The input tensor must be a sparse tensor.")
    
    # Extract the indices and values
    indices = sparse_tensor._indices()
    values = sparse_tensor._values()
    
    old_size = sparse_tensor.size()
    new_num_elements = new_size[-1]

    if old_size[0] * old_size[1] != new_num_elements:
        raise ValueError("The total number of elements must match between the old and new sizes.")
    
    # Convert 2D indices to 1D indices for the new shape
    if len(old_size) == 2 and old_size[0] == new_size[1] // old_size[1]:
        row_indices = indices[0]
        col_indices = indices[1]
        linear_indices = row_indices * old_size[1] + col_indices
        new_indices = torch.stack([torch.zeros_like(linear_indices), linear_indices], dim=0)
    else:
        raise ValueError("Unsupported resize operation.")
    
    # Create a new sparse tensor with the same non-zero elements but a different size
    new_sparse_tensor = torch.sparse_coo_tensor(new_indices, values, new_size, dtype=sparse_tensor.dtype).coalesce()
    
    return new_sparse_tensor




def sparse_one_hot(indices, num_classes):
    """
    Creates a sparse one-hot encoding for given indices.

    Args:
        indices: A tensor of shape (batch_size,) containing the indices to be one-hot encoded.
        num_classes: The number of classes (i.e., the size of the one-hot encoding).

    Returns:
        A sparse tensor of shape (batch_size, num_classes) with one-hot encoding.
    """
    batch_size = indices.size(0)
    rows = torch.arange(batch_size)
    indices = torch.stack([rows, indices], dim=0)
    values = torch.ones(batch_size)
    
    return torch.sparse_coo_tensor(indices, values, (batch_size, num_classes))


def concatenate_sparse_tensors(sparse_tensors, dim=0):
    """
    Concatenate a list of sparse tensors along a specified dimension.

    Args:
        sparse_tensors (list of torch.sparse.FloatTensor): List of sparse tensors to concatenate.
        dim (int): Dimension along which to concatenate the tensors.

    Returns:
        torch.sparse.FloatTensor: Concatenated sparse tensor.
    """
    if not all(tensor.is_sparse for tensor in sparse_tensors):
        raise ValueError("All tensors must be sparse.")

    indices_list = []
    values_list = []
    sizes = [list(tensor.size()) for tensor in sparse_tensors]

    if dim < 0:
        dim = sparse_tensors[0].dim() + dim

    for i, tensor in enumerate(sparse_tensors):
        indices = tensor._indices()
        values = tensor._values()
        
        if dim > 0:
            offset = sum(size[dim] for size in sizes[:i])
            indices[dim] += offset
        
        indices_list.append(indices)
        values_list.append(values)

    concatenated_indices = torch.cat(indices_list, dim=1)
    concatenated_values = torch.cat(values_list)

    concatenated_size = list(sizes[0])
    concatenated_size[dim] = sum(size[dim] for size in sizes)

    return torch.sparse_coo_tensor(concatenated_indices, concatenated_values, size=concatenated_size)

def trajectory_balance_loss(total_flow, rewards, fwd_probs, back_probs):
    """
    Computes the mean trajectory balance loss for a collection of samples. For
    more information, see Bengio et al. (2022): https://arxiv.org/abs/2201.13259
    
    Args:
        total_flow: The estimated total flow used by the GFlowNet when drawing
        the collection of samples for which the loss should be computed
        
        rewards: The rewards associated with the final state of each of the
        samples
        
        fwd_probs: The forward probabilities associated with the trajectory of
        each sample (i.e. the probabilities of the actions actually taken in
        each trajectory)
        
        back_probs: The backward probabilities associated with each trajectory
    """
    eps = 1e-9  # Small epsilon to avoid log(0)

    # Ensure all tensors are in the same device and dtype
    total_flow = total_flow.to(fwd_probs.dtype)
    rewards = rewards.to(fwd_probs.dtype)
    back_probs = back_probs.to(fwd_probs.dtype)

    # Calculate the forward log probabilities
    log_fwd_probs = torch.log(fwd_probs + eps)  # Adding a small value to avoid log(0)
    
    # Calculate the backward log probabilities
    log_back_probs = torch.log(back_probs + eps)  # Adding a small value to avoid log(0)
    
    # Sum log probabilities along the trajectory
    log_fwd_probs_sum = log_fwd_probs.sum(dim=-1)
    log_back_probs_sum = log_back_probs.sum(dim=-1)
    
    # Use log-sum-exp trick for numerical stability
    max_log_fwd = log_fwd_probs_sum.max(dim=0, keepdim=True)[0]
    max_log_back = log_back_probs_sum.max(dim=0, keepdim=True)[0]
    
    # Normalize log probabilities to prevent numerical underflow
    normalized_log_fwd = log_fwd_probs_sum - max_log_fwd
    normalized_log_back = log_back_probs_sum - max_log_back
    
    # Compute lhs and rhs in the log domain
    log_lhs = torch.log(total_flow + eps) + normalized_log_fwd
    log_rhs = torch.log(rewards + eps) + normalized_log_back
    
    # Compute the trajectory balance loss
    loss = (log_lhs - log_rhs)**2
    
    return loss.mean()

def log_memory_usage(stage: str):
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"[{stage}] CPU Memory Usage: {mem_info.rss / (1024 ** 2):.2f} MB; VMS: {mem_info.vms / (1024 ** 2):.2f} MB")
    if torch.cuda.is_available():
        print(f"[{stage}] GPU Memory Usage: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")

def malloc_usage(description):
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    print(f"\nMemory usage at {description}:")
    for stat in top_stats:  # Print top 10 lines
        print(stat)

def update_edges_and_convert_to_sparse(data: Data, actions: List[Tensor], matrix_size: int) -> torch.sparse.FloatTensor:
    """
    Removes edges from the PyTorch Geometric Data object based on the actions and converts
    the resulting graph into a sparse tensor of size (matrix_size, matrix_size).
    
    Args:
        data (Data): The PyTorch Geometric data object representing the graph.
        actions (List[int]): The list of actions (edge indices) to be removed from the graph.
        matrix_size (int): The size of the matrix (assumed to be square: matrix_size * matrix_size).

    Returns:
        torch.sparse.FloatTensor: The resulting sparse tensor after removing edges.
    """
    
    # Get the edge_index and edge_attr from the PyTorch Geometric data object
    edge_index = data.edge_index  # Shape: [2, num_edges]
    edge_attr = data.edge_attr if 'edge_attr' in data else None  # Optional edge attributes

   #print(f"Original number of edges: {edge_index.size(1)}")

    actions = [int(action.item()) for action in actions]

    # Convert actions to a set for fast look-up
    actions_set = set(actions)

    #print(f"Actions to remove: {actions_set}")

    # Filter out the edges indicated by the actions (remove the edges corresponding to the actions)
    remaining_edges_mask = [i for i in range(edge_index.size(1)) if i not in actions_set]

    if len(remaining_edges_mask) == 0:
       print(f"No edges left after filtering. Returning an empty sparse tensor.")

   #print(f"Remaining edges after filtering: {remaining_edges_mask}")
   #print(f"Remaining number of edges: {len(remaining_edges_mask)}")

    remaining_edge_index = edge_index[:, remaining_edges_mask]
    
    # If there are edge attributes, filter them as well
    if edge_attr is not None:
        remaining_edge_attr = edge_attr[remaining_edges_mask]
    else:
        remaining_edge_attr = torch.ones(remaining_edge_index.size(1))  # Default to all 1s for remaining edges

   #print(f"Remaining edge index:\n{remaining_edge_index}")
   #print(f"Remaining edge attributes:\n{remaining_edge_attr}")

    # Flatten the edge indices into a 1D array suitable for a matrix representation (row * matrix_size + col)
    row, col = remaining_edge_index
    flattened_indices = row * matrix_size + col

   #print(f"Flattened indices:\n{flattened_indices}")
    # Create the sparse tensor of size (matrix_size * matrix_size)
    sparse_tensor = torch.sparse_coo_tensor(
        indices=torch.stack([torch.zeros_like(flattened_indices), flattened_indices]),  # Add a zero dimension for batch size
        values=remaining_edge_attr.float(),  # Edge values (1 by default or based on edge_attr)
        size=(1, matrix_size * matrix_size),
        dtype=torch.float32
    ).coalesce()
   #print(f"Sparse Tensor non-zeros {sparse_tensor._nnz()}")

    return sparse_tensor

def matrix_flops(matrix: Tensor) -> Tuple[int, int]:
    if matrix.is_sparse:
        # For sparse tensors, the number of non-zero elements is the size of the values tensor
        non_zeros = matrix._values().numel()
        flops = non_zeros * matrix.shape[1] * 2
    else:
        # We should be working exclusively with sparse tensors, so maybe replace this if/else statement with an assert. 
        non_zeros = torch.nonzero(matrix).size(0)
        flops = 2 * non_zeros        
    return flops, non_zeros

def calculate_residual(updated_matrix: Tensor, original_matrix: Tensor) -> Tensor:
    #Calculate residual term, moving to its own method because we only want to calculate ||A*A-I|| once
    # Residual term: ||M*A - I||
    i = torch.arange(0, original_matrix.size(0))
    i = torch.stack([i, i])
    v = torch.ones(original_matrix.size(0), dtype=torch.float64)
    sparse_identity = torch.sparse_coo_tensor(i, v, (original_matrix.size(0), original_matrix.size(0)))
    product = torch.mm(updated_matrix, original_matrix)
    #print(f"Product {product}")
    residual = torch.norm(product - sparse_identity)
    #print(f"residual {residual}")

    return residual

def evaluate_preconditioner(updated_matrix: Tensor, original_matrix: Tensor, orig_residual: float, orig_flops: int, alpha: float) -> float:

    
    # Compute the computational cost (floating point operations - FLOPs)
    # For simplicity, we can assume 2 FLOPs for each non-zero element (one for multiplication and one for addition)
    # in the original matrix when multiplied by the preconditioner
    # Assuming 'matrix' is the sparse matrix
    
    residual = calculate_residual(updated_matrix, original_matrix)
    #print(f"Updated Matrix Flops Shape: {updated_matrix.shape}")
    #print(f"Updated Matrix NNZ {updated_matrix._nnz()}")
    flops, non_zeros = matrix_flops(updated_matrix)
    
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
    
    performance_metric = alpha * (1 - residual_ratio) + (1 - alpha) * (1 - computational_ratio)
    #print(f"performance metric {performance_metric}")
    return performance_metric

def calculate_reward(starting_matrix, updated_matrix, orig_residual, orig_flops, traj_length: int, alpha: float) -> float:
    # Use the current matrix as a preconditioner and calculate the reward
    # based on its performance (e.g., reduced iterations, improved stability)
    #Need to figure out a way to penalize a trajectory to avoid the model picking a blank matrix. For now, dividing by log length of trajectory, but will need to come up with something.
    #print(f"Reward Method before Traj Length: {reward}")
    # Compute the computational cost (floating point operations - FLOPs)
    # For simplicity, we can assume 2 FLOPs for each non-zero element (one for multiplication and one for addition)
    # in the original matrix when multiplied by the preconditioner
    # Assuming 'matrix' is the sparse matrix
    #print(f"SM Type {type(starting_matrix)}")
    residual = calculate_residual(updated_matrix, starting_matrix)
    #print(f"Updated Matrix Flops Shape: {updated_matrix.shape}")
    #print(f"Updated Matrix NNZ {updated_matrix._nnz()}")
    flops, non_zeros = matrix_flops(updated_matrix)
    
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
    
    reward = alpha * (1 - residual_ratio) + (1 - alpha) * (1 - computational_ratio)
    #reward = reward/torch.log(torch.tensor(traj_length, dtype=torch.float64))
    #reward = (reward/torch.tensor(traj_length, dtype=torch.float64)) * 100
    reward = reward.to(torch.float64) * 1000
    #print(f"Reward Method after Traj Length: {reward}")
    return reward