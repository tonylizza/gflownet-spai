import torch
from torch import Tensor
from torch_geometric.data import Data
import scipy.io

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

        print(f"End start: {end, start}")
        print(f"New indices shape {new_indices.shape}")
        print(f"New values shape {new_values}")

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
    total_flow = total_flow.to(fwd_probs.device).to(fwd_probs.dtype)
    rewards = rewards.to(fwd_probs.device).to(fwd_probs.dtype)
    back_probs = back_probs.to(fwd_probs.device).to(fwd_probs.dtype)

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
