import torch
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
from scipy.io import mmread
from scipy.sparse.linalg import gmres, spilu, LinearOperator
from scipy.sparse import csr_matrix, csc_matrix
import time

def validate_gflownet(model, validation_data_module, max_iters=100):
    """
    This function runs validation on the trained GFlowNet model.
    
    Args:
        model: Trained GFlowNet model.
        validation_data_module: Data module containing validation data.
        max_iters: Maximum number of iterations for GMRES (or other solver).
        
    Returns:
        Validation results: A dictionary containing results such as GMRES iterations, time, etc.
    """
    model.eval()  # Set the model to evaluation mode
    results = []

    # Disable gradient computation for validation
    with torch.no_grad():
        for batch in tqdm(validation_data_module.test_dataloader()):
            # Generate the sparse matrix via the trained policy
            sampled_trajectory = model.sample_trajectory(batch)
            
            # Apply your GMRES solver on the original ILU preconditioner
            original_iterations, original_time = solve_with_gmres(batch['starting_matrix'], max_iters=max_iters)

            # Apply GMRES on the matrix with entries removed by the model's sampling
            sparse_iterations, sparse_time = solve_with_gmres(sampled_trajectory, max_iters=max_iters)

            # Store the results for comparison
            results.append({
                "original_iterations": original_iterations,
                "original_time": original_time,
                "sparse_iterations": sparse_iterations,
                "sparse_time": sparse_time,
            })
    
    return results


# Function to load matrix A from .mtx file
def load_mtx_file(file_path):
    matrix = mmread(file_path)
    return csr_matrix(matrix)

def load_mtx_file_csc(file_path):
    matrix = mmread(file_path)
    return csc_matrix(matrix)

def load_vector_mtx(file_path):
    vector = mmread(file_path)  # Load the vector (could be sparse or dense)
    
    # Check if the loaded data is a sparse matrix, if so convert it to a dense array
    if hasattr(vector, "toarray"):
        vector = vector.toarray()
    
    # Flatten the array if it's a row or column vector
    vector = np.array(vector)
    vector = vector.flatten()
    
    return vector

# Function to solve the system using GMRES with an optional preconditioner
def solve_with_gmres(A, b, M=None, max_iters=20000):
    # Ensure b is a 1D array with the same number of rows as A
    b = b.flatten()
    if b.shape[0] != A.shape[0]:
        raise ValueError(f"Shape mismatch: A is {A.shape}, but b is {b.shape}")
    
    # Initial guess (zero vector)
    x0 = np.zeros(b.shape)

    # Lists to store iteration number and residual norm
    residuals = []
    
    # Callback function to capture residual norm at each iteration
    def callback(rk):
        print(rk)
        residuals.append(rk)
    
    # Measure computational time
    start_time = time.time()
    
    # Use GMRES to solve the system Ax = b with preconditioner M
    x, exitCode = gmres(A, b, x0=x0, M=M, maxiter=max_iters, callback=callback)
    
    elapsed_time = time.time() - start_time
    
    if exitCode == 0:
        print("GMRES converged successfully.")
    else:
        print(f"GMRES did not converge. Exit code: {exitCode}")
    
    # Number of iterations is the length of the residuals list
    num_iterations = len(residuals)
    
    return x, residuals, num_iterations, elapsed_time
