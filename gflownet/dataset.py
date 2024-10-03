import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from torch_geometric.data import Data
from scipy.io import mmread  # assuming your matrices are in Matrix Market format
import pytorch_lightning as pl
import numpy as np
import os
from gflownet.utils import matrix_flops, calculate_residual
from gflownet.validate import load_mtx_file, load_vector_mtx
from scipy.sparse import csr_matrix, csc_matrix


class SparseMatrixDataset(Dataset):
    def __init__(self, ilu_paths, orig_matrix_paths, b_vector_paths, load_b=True):
        """
        :param ilu_paths: List of paths to ILU preconditioner matrix files.
        :param orig_matrix_paths: List of paths to original matrix files.
        :param b_vector_paths: List of paths to 'b' vector files.
        :param load_b: Whether to load 'b' vectors (used in validation).
        """
        self.ilu_paths = ilu_paths
        self.orig_matrix_paths = orig_matrix_paths
        self.b_vector_paths = b_vector_paths
        self.load_b = load_b

        # Check if we have the same number of ILU matrices, original matrices, and b vectors
        assert len(self.ilu_paths) == len(self.orig_matrix_paths) == len(self.b_vector_paths), \
            "Mismatch in number of ILU matrices, original matrices, and b vectors."

    def __len__(self):
        return len(self.ilu_paths)

    def __getitem__(self, idx):
        # Load ILU preconditioner matrix for training
        ilu_matrix_path = self.ilu_paths[idx]
        ilu_matrix = mmread(ilu_matrix_path).tocoo()
        ilu_values = torch.tensor(ilu_matrix.data, dtype=torch.float32)
        ilu_indices = torch.tensor([ilu_matrix.row, ilu_matrix.col], dtype=torch.long)
        ilu_sparse_tensor = torch.sparse_coo_tensor(ilu_indices, ilu_values, ilu_matrix.shape)
        ilu_csr = load_mtx_file(ilu_matrix_path)
        #print(f"Shape of ilu_csr {ilu_csr.shape}")
        
        # Create PyTorch Geometric `Data` object for ILU preconditioner
        edge_index = torch.stack([torch.tensor(ilu_matrix.row, dtype=torch.long), torch.tensor(ilu_matrix.col, dtype=torch.long)], dim=0)
        edge_attr = torch.tensor(ilu_matrix.data, dtype=torch.float32)
        x = torch.ones((ilu_matrix.shape[0], 1))  # Example node features
        
        ilu_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        matrix_sq_side = ilu_sparse_tensor.size(0)

        starting_flops, _ = matrix_flops(ilu_sparse_tensor)
        starting_residual = calculate_residual(ilu_sparse_tensor, ilu_sparse_tensor)

        # Load original matrix for evaluation (if necessary)
        orig_matrix = mmread(self.orig_matrix_paths[idx]).tocoo()
        orig_csr = load_mtx_file(self.orig_matrix_paths[idx]).tocsc()
        orig_sparse_tensor = torch.sparse_coo_tensor(
            torch.tensor([orig_matrix.row, orig_matrix.col], dtype=torch.long),
            torch.tensor(orig_matrix.data, dtype=torch.float32),
            orig_matrix.shape
        )

        # Load 'b' vector for evaluation
        b_vector = None
        if self.load_b:
            b_vector = load_vector_mtx(self.b_vector_paths[idx])
            print(f"Type of b_vector: {type(b_vector)}")

        return {
            'ilu_data': ilu_data,
            'starting_flops': starting_flops,
            'starting_residual': starting_residual,
            'matrix_sq_side': matrix_sq_side,
            'starting_matrix': ilu_sparse_tensor,
            'starting_csr': ilu_csr,
            'original_matrix': orig_sparse_tensor,
            'original_csr': orig_csr,
            'b_vector': b_vector
        }


class MatrixDataModule(pl.LightningDataModule):
    def __init__(self, matrix_directory, batch_size=1, load_b=True):
        super().__init__()
        self.matrix_directory = matrix_directory
        self.batch_size = batch_size
        self.load_b = load_b

    def setup(self, stage=None):
        # Get all filenames (without extensions) in the directory
        filenames = sorted([os.path.splitext(f)[0].replace('_ilu', '') for f in os.listdir(os.path.join(self.matrix_directory, 'ilu_matrices')) if f.endswith('_ilu.mtx')])

        # Paths to ILU matrices, original matrices, and b vectors
        self.ilu_paths = [os.path.join(self.matrix_directory, 'ilu_matrices', f"{filename}_ilu.mtx") for filename in filenames]
        self.orig_matrix_paths = [os.path.join(self.matrix_directory, 'matrices', f"{filename}.mtx") for filename in filenames]
        self.b_vector_paths = [os.path.join(self.matrix_directory, 'b_vectors', f"{filename}_b.mtx") for filename in filenames]

        # Split into training and validation sets (80/20 split)
        split_index = int(0.8 * len(self.ilu_paths))
        self.train_ilu_paths = self.ilu_paths[:split_index]
        self.val_ilu_paths = self.ilu_paths[split_index:]
        self.train_orig_paths = self.orig_matrix_paths[:split_index]
        self.val_orig_paths = self.orig_matrix_paths[split_index:]
        self.train_b_paths = self.b_vector_paths[:split_index]
        self.val_b_paths = self.b_vector_paths[split_index:]

    def train_dataloader(self):
        # Create dataset and dataloader for training
        train_dataset = SparseMatrixDataset(self.train_ilu_paths, self.train_orig_paths, self.train_b_paths, load_b=False)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=custom_collate)

    def val_dataloader(self):
        # Create dataset and dataloader for validation
        val_dataset = SparseMatrixDataset(self.val_ilu_paths, self.val_orig_paths, self.val_b_paths, load_b=self.load_b)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=custom_collate)


def custom_collate(batch):
    # Separate the PyTorch Geometric `Data` objects and other custom data
    data_list = [item['ilu_data'] for item in batch]  # Extract the `Data` objects for ILU matrices
    
    # Convert scalar values to tensors and keep sparse tensors in their format
    other_values = {}
    for key in batch[0]:
        if key != 'ilu_data':
            if isinstance(batch[0][key], (int, float)):
                # Handle scalar values (e.g., matrix_sq_side, starting_flops, etc.)
                other_values[key] = torch.tensor([item[key] for item in batch])
            elif isinstance(batch[0][key], torch.Tensor) and batch[0][key].is_sparse:
                # Handle sparse tensors (e.g., starting_matrix)
                other_values[key] = batch[0][key]  # Ensuring sparse tensors are in the correct format
            elif isinstance(batch[0][key], torch.Tensor):
                # Handle other dense tensors
                other_values[key] = torch.stack([item[key] for item in batch])
            elif isinstance(batch[0][key], (np.ndarray, csr_matrix, csc_matrix)):  # Check for SciPy csr_matrix and NumPy arrays
                # Keep NumPy arrays and SciPy sparse matrices (e.g., ilu_csr and orig_csr) as-is
                other_values[key] = batch[0][key]
            else:
                # Keep as a list if not a tensor
                other_values[key] = [item[key] for item in batch]

    # Use PyTorch Geometric's Batch class to handle the `Data` objects
    batch_data = Batch.from_data_list(data_list)

    # Return the batch containing the batched `Data` and other tensor values
    return {'data': batch_data, **other_values}
