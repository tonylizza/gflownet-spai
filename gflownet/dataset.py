import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from torch_geometric.data import Data
from scipy.io import mmread  # assuming your matrices are in Matrix Market format
import pytorch_lightning as pl
import os
from gflownet.utils import matrix_flops, calculate_residual

class SparseMatrixDataset(Dataset):
    def __init__(self, matrix_paths):
        self.matrix_paths = matrix_paths  # List of paths to matrix files

    def __len__(self):
        return len(self.matrix_paths)

    def __getitem__(self, idx):
        # Lazy load the matrix when it's requested
        matrix_path = self.matrix_paths[idx]
        matrix = mmread(matrix_path).tocoo()  # Load matrix from file in COOrdinate format
        values = torch.tensor(matrix.data, dtype=torch.float32)
        indices = torch.tensor([matrix.row, matrix.col], dtype=torch.long)
        sparse_tensor = torch.sparse_coo_tensor(indices, values, matrix.shape)

        matrix_sq_side = sparse_tensor.size(0)
        
        # Create PyTorch Geometric `Data` object
        edge_index = torch.stack([torch.tensor(matrix.row, dtype=torch.long), torch.tensor(matrix.col, dtype=torch.long)], dim=0)
        edge_attr = torch.tensor(matrix.data, dtype=torch.long)
        x = torch.ones((matrix.shape[0], 1))  # Example node features
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        starting_flops, _ = matrix_flops(sparse_tensor)

        #Calculate self-residual for starting matrix as part of data load

        starting_residual = calculate_residual(sparse_tensor, sparse_tensor)
        
        return {'data': data, 'starting_flops': starting_flops, 'starting_residual': starting_residual, 'matrix_sq_side': matrix_sq_side, 'starting_matrix': sparse_tensor}


class MatrixDataModule(pl.LightningDataModule):
    def __init__(self, matrix_directory, batch_size=1):
        super().__init__()
        self.matrix_directory = matrix_directory
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Load all matrix file paths
        self.matrix_paths = [os.path.join(self.matrix_directory, f) for f in os.listdir(self.matrix_directory) if f.endswith('.mtx')]
        print(f"{self.matrix_paths}")

    def train_dataloader(self):
        # Create dataset and dataloader
        dataset = SparseMatrixDataset(self.matrix_paths)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=custom_collate)


def custom_collate(batch):
    # Separate the PyTorch Geometric `Data` objects and other custom data
    data_list = [item['data'] for item in batch]  # Extract the `Data` objects
    
    # Convert scalar values to tensors and keep sparse tensors in their format
    other_values = {}
    for key in batch[0]:
        if key != 'data':
            if isinstance(batch[0][key], (int, float)):
                # Handle scalar values (e.g., matrix_sq_side, starting_flops, etc.)
                other_values[key] = torch.tensor([item[key] for item in batch])
            elif isinstance(batch[0][key], torch.Tensor) and batch[0][key].is_sparse:
                # Handle sparse tensors (e.g., starting_matrix)
                other_values[key] = batch[0][key]  # Ensuring sparse tensors are in the correct format
            elif isinstance(batch[0][key], torch.Tensor):
                # Handle other dense tensors
                other_values[key] = torch.stack([item[key] for item in batch])
            else:
                # Keep as a list if not a tensor
                other_values[key] = [item[key] for item in batch]

    # Use PyTorch Geometric's Batch class to handle the `Data` objects
    batch_data = Batch.from_data_list(data_list)

    # Return the batch containing the batched `Data` and other tensor values
    return {'data': batch_data, **other_values}


