# %%
import torch
from torch.nn.functional import one_hot
from torch.optim import Adam
import torch.optim as optim

from tqdm import tqdm
from preconditioner import PreconditionerEnv
from policy import ForwardPolicy, BackwardPolicy
from gflownet.gflownet import GFlowNet
from gflownet.utils import sparse_one_hot
from gflownet.utils import trajectory_balance_loss, market_matrix_to_sparse_tensor, log_memory_usage, malloc_usage
import psutil

# %%
print(torch.__version__)

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
#filename = 'LF10' # 18x18 matrix
#filename = 'bcsstk03' #100x100 matrix
filename = 'olm500' #500x500 matrix

# %%
data_directory = 'data/' + filename + '/'

# %%
matrix_path = data_directory + filename + '.mtx'  # Update this with your file path
batch_size = 2
num_epochs = 1000
lr = 0.0005

# %%
import numpy as np
from scipy.io import mmread
from scipy.sparse.linalg import gmres, spilu, LinearOperator
from scipy.sparse import csr_matrix
import time

# Function to load matrix A from .mtx file
def load_mtx_file(file_path):
    matrix = mmread(file_path)
    return csr_matrix(matrix)

def load_vector_mtx(file_path):
    vector = mmread(file_path)  # Load the vector (could be sparse or dense)
    
    # Check if the loaded data is a sparse matrix, if so convert it to a dense array
    if hasattr(vector, "toarray"):
        vector = vector.toarray()
    
    # Flatten the array if it's a row or column vector
    vector = vector.flatten()
    
    return vector

# Function to solve the system using GMRES with an optional preconditioner
def solve_with_gmres(A, b, M=None):
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
        residuals.append(rk)
    
    # Measure computational time
    start_time = time.time()
    
    # Use GMRES to solve the system Ax = b with preconditioner M
    x, exitCode = gmres(A, b, x0=x0, M=M, maxiter=10260, callback=callback)
    
    elapsed_time = time.time() - start_time
    
    if exitCode == 0:
        print("GMRES converged successfully.")
    else:
        print(f"GMRES did not converge. Exit code: {exitCode}")
    
    # Number of iterations is the length of the residuals list
    num_iterations = len(residuals)
    
    return x, residuals, num_iterations, elapsed_time


# %%
# Example usage
mtx_file_path_A = matrix_path  # Replace with your actual matrix file path
#mtx_file_path_b = data_directory + 'hangGlider_3_b.mtx'  # Replace with your actual vector file path

# Load the vector data as a numpy array
#b = mmread(mtx_file_path_b)



# Load A and b from the .mtx files
A = load_mtx_file(mtx_file_path_A)

# %%
log_memory_usage("Before Loading Initial Matrix")

# Load the initial matrix from a file
original_matrix = market_matrix_to_sparse_tensor(matrix_path)

log_memory_usage("After Loading Initial Matrix")


# %%
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import gmres
import tracemalloc  # For tracking memory usage
#Sparse ILU to create baseline preconditioner

# Compute the ILU factorization
ilu = spla.spilu(A)

# Define a function to apply the ILU preconditioner
M_x = lambda x: ilu.solve(x)

# Create a LinearOperator object from the ILU solver function
M = spla.LinearOperator(A.shape, M_x)

# %%
#Convert SuperLU object into LU sparse tensor
# Extract L and U from the ILU factorization (spilu)
L = sp.tril(ilu.L, format='csr')  # Lower triangular matrix from ILU
U = sp.triu(ilu.U, format='csr')  # Upper triangular matrix from ILU

# Multiply L and U to form the combined LU matrix
LU = L @ U  # Sparse matrix multiplication to maintain sparsity

# Convert the LU matrix to a PyTorch sparse tensor
coo = LU.tocoo()  # Convert to COO format for PyTorch compatibility
values = coo.data
indices = np.vstack((coo.row, coo.col))

i = torch.LongTensor(indices)
v = torch.FloatTensor(values)
shape = torch.Size(coo.shape)

#Initial matrix to sample for model
initial_matrix = torch.sparse_coo_tensor(i, v, shape)

# Print information about the PyTorch sparse tensor
print(f"PyTorch sparse tensor shape: {initial_matrix.shape}")
print(f"Number of non-zero elements: {initial_matrix._nnz()}")
print(f"Indices: {initial_matrix._indices()}")
print(f"Values: {initial_matrix._values()}")

# %% [markdown]
# # Structured Sampling Preconditioner

# %%
#initial_matrix = structured_sampling(original_matrix, 4, 0.75)
matrix_size = initial_matrix.size(0)

# %%
original_matrix.dtype

# %%
# Initialize the environment and policies
env = PreconditionerEnv(matrix_size=matrix_size, initial_matrix=initial_matrix, original_matrix=initial_matrix)
env.data.edge_attr.shape

# %%

node_features = -1
input_dim = 1
hidden_dim = 4
forward_policy = ForwardPolicy(node_features=node_features, hidden_dim=hidden_dim)
#forward_policy = ForwardPolicy(node_features=node_features, hidden_dim=hidden_dim, num_actions=env.num_actions)
backward_policy = BackwardPolicy(input_dim=input_dim, hidden_dim=hidden_dim)

# %%
env.data.edge_attr.shape

# %%
initial_matrix.size()

# %%
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                print(f"{name}: {param.grad.norm()}")
            else:
                print(f"{name}: No gradient")


# %%
log_memory_usage("Before Starting Training")

# %%
import pandas as pd

def capture_training_data(epoch, log, loss, batch_size, report_data=None, detailed_report_data=None):
    """
    Captures and returns the report data and detailed report data from training logs.

    Parameters:
    - epoch (int): The current epoch number.
    - log (object): An object containing training logs with `_actions` and `rewards`.
    - loss (torch.Tensor): The loss value for the current epoch.
    - batch_size (int): The number of samples in the batch.
    - report_data (pd.DataFrame, optional): Existing report DataFrame to be updated. Defaults to None.
    - detailed_report_data (pd.DataFrame, optional): Existing detailed report DataFrame to be updated. Defaults to None.

    Returns:
    - report_data (pd.DataFrame): Updated report DataFrame.
    - detailed_report_data (pd.DataFrame): Updated detailed report DataFrame.
    """
    
    # Initialize DataFrames if not provided
    if report_data is None:
        report_data = pd.DataFrame(columns=['epoch', 'num_actions', 'loss', 'reward'])
    if detailed_report_data is None:
        detailed_report_data = pd.DataFrame(columns=['epoch', 'sample_number', 'num_actions', 'loss', 'reward'])

    # Capture summary data
    total_length = len(log._actions)
    new_report_data = pd.DataFrame([{
        'epoch': epoch,
        'num_actions': total_length,
        'loss': loss.item(),
        'reward': log.rewards
    }])
    report_data = pd.concat([report_data, new_report_data], ignore_index=True)
    
    # Capture detailed data for each sample in the batch
    batch_data = []
    for sample_id in range(batch_size):
        sum_actions = log._actions.t()[sample_id]
        mask_actions = sum_actions != -1
        num_actions = mask_actions.sum()
        reward = log.rewards[sample_id].item() if isinstance(log.rewards, torch.Tensor) else log.rewards[sample_id]
        batch_data.append({
            'epoch': epoch,
            'sample_number': sample_id + 1,  # Sample number within the batch/epoch
            'num_actions': num_actions.item(),
            'loss': loss.item(),
            'reward': reward
        })
    
    detailed_report_data = pd.concat([detailed_report_data, pd.DataFrame(batch_data)], ignore_index=True)

    return report_data, detailed_report_data


# %%
import pandas as pd

tracemalloc.start()
# Initialize the GFlowNet model
model = GFlowNet(forward_policy, backward_policy, env)
opt = Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.2, patience=10, verbose=True)


log_memory_usage("After Model Initialization")

report_data = pd.DataFrame(columns=['epoch', 'num_actions', 'loss', 'reward'])

detailed_report_data = pd.DataFrame(columns=['epoch', 'sample_number', 'num_actions', 'loss', 'reward'])

s0 = [initial_matrix.clone() for _ in range(batch_size)]

for epoch in (p := tqdm(range(num_epochs))):
    #malloc_usage(f"Start of Epoch {epoch}")

    model.train()
    #opt.zero_grad()

    # Initialize the starting states
    initial_indices = torch.zeros(batch_size).long()
    #s0 = [sparse_one_hot(initial_indices[i:i+1], env.state_dim).float() for i in range(batch_size)]
    #print(f"Cloned initial matrix")
    #s0 = one_hot(torch.zeros(batch_size).long(), env.state_dim).float()
    # Sample final states and log information
    #malloc_usage("Before Sample States")
    log = model.sample_states(s0, return_log=True)
    
    # Calculate the trajectory balance loss
    loss = trajectory_balance_loss(log.total_flow,
                                    log.rewards,
                                    log.fwd_probs,
                                    log.back_probs)
    
    #print(f"log.total_flow {log.total_flow}")
    #print(f"log.rewards {log.rewards}")
    #print(f"log.fwd_probs {len(log.fwd_probs)}")
    #print(f"log.back_probs {len(log.back_probs)}")
    #print(f"log._actions shape {len(log._actions)}")
    #print(f"Loss Calculation: {loss}")
    # Backpropagation and optimization step
    # Check for NaN or Inf in loss before proceeding
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"Warning: NaN or Inf detected in loss at epoch {epoch}. Skipping backpropagation.")
        continue  # Skip the current iteration if NaN or Inf is detected
    scheduler.step(loss)
    loss.backward()
    #check_gradients(model)
    opt.step()
    #named_params = model.named_parameters()
    opt.zero_grad()
    #Capture data
    total_length = len(log._actions)
    #report_data, detailed_report_data = capture_training_data(epoch=0, log=log, loss=loss, batch_size=batch_size, report_data=report_data, detailed_report_data=detailed_report_data)
    
    if epoch % 10 == 0:
       tqdm.write(f"Epoch {epoch} Loss: {loss.item():.3f}, Num_Actions {total_length}")
       #malloc_usage(f"At {epoch} Epochs")



# %%
report_data.to_csv('training_log.csv', index=False)

# %%
detailed_report_data.to_csv('detailed_training_log.csv', index=False)

# %%
import plotly.graph_objects as go
# Extract the data
epochs = report_data['epoch'].values
num_actions = report_data['num_actions'].values
losses = report_data['loss'].values

# Extract the data
epochs = report_data['epoch'].values
num_actions = report_data['num_actions'].values
losses = report_data['loss'].values

# Create the 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=epochs,
    y=num_actions,
    z=losses,
    mode='markers',
    marker=dict(
        size=5,
        color=losses,
        colorscale='Viridis',
        opacity=0.8
    ),
    text=[f'Epoch: {e}<br>Num Actions: {n}<br>Loss: {l}' for e, n, l in zip(epochs, num_actions, losses)],
    hoverinfo='text'
)])

# Update the layout
fig.update_layout(
    scene=dict(
        xaxis=dict(
            title='Epoch',
            range=[0, max(epochs) * 1.1]  # Extend the range slightly beyond the max epoch
        ),
        yaxis=dict(
            title='Number of Actions'
        ),
        zaxis=dict(
            title='Loss'
        )
    ),
    width=1000,
    height=800
)

# Show the plot
fig.show()

# %%
# Extract the data
epochs = report_data['epoch'].values
losses = report_data['loss'].values

# Create the 2D scatter plot
fig = go.Figure(data=go.Scatter(
    x=epochs,
    y=losses,
    mode='lines+markers',
    marker=dict(
        size=5,
        color='blue'
    ),
    text=[f'Epoch: {e}<br>Loss: {l}' for e, l in zip(epochs, losses)],
    hoverinfo='text'
))

# Update the layout
fig.update_layout(
    xaxis=dict(
        title='Epoch'
    ),
    yaxis=dict(
        title='Loss'
    ),
    width=1000,
    height=600,
    title='Epoch vs Loss'
)

# Show the plot
fig.show()

# %%
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

# Extract the data
epochs = report_data['epoch'].values.reshape(-1, 1)
losses = report_data['loss'].values

# Perform linear regression
reg = LinearRegression().fit(epochs, losses)
slope = reg.coef_[0]
intercept = reg.intercept_

# Calculate the regression line
regression_line = reg.predict(epochs)

# Create the 2D scatter plot
fig = go.Figure()

# Add the original data
fig.add_trace(go.Scatter(
    x=report_data['epoch'],
    y=report_data['loss'],
    mode='markers',
    marker=dict(
        size=5,
        color='blue'
    ),
    name='Loss',
    text=[f'Epoch: {e}<br>Loss: {l}' for e, l in zip(report_data['epoch'], report_data['loss'])],
    hoverinfo='text'
))

# Add the regression line
fig.add_trace(go.Scatter(
    x=report_data['epoch'],
    y=regression_line,
    mode='lines',
    line=dict(
        color='red'
    ),
    name='Regression Line'
))

# Update the layout
fig.update_layout(
    xaxis=dict(
        title='Epoch'
    ),
    yaxis=dict(
        title='Loss'
    ),
    width=1000,
    height=600,
    title=f'Epoch vs Loss (Slope: {slope:.4f})'
)

# Show the plot
fig.show()

# Print the slope to determine the trend
print(f"The slope of the regression line is {slope:.4f}")
if slope < 0:
    print("The values are trending down.")
elif slope > 0:
    print("The values are trending up.")
else:
    print("The values are constant.")

# %% [markdown]
# 

# %%
# Function to check for duplicates across columns
def find_column_duplicates(tensor, check_value=None):
    num_columns = tensor.size(1)
    duplicates = {}
    check_value_duplicates = {}
    
    for col in range(num_columns):
        seen = set()
        col_duplicates = set()
        for row in range(tensor.size(0)):
            value = tensor[row, col].item()
            if value in seen:
                col_duplicates.add(value)
            seen.add(value)
        
        if col_duplicates:
            duplicates[col] = col_duplicates
        
        if check_value is not None and check_value in seen:
            check_value_duplicates[col] = check_value in col_duplicates
    
    return duplicates, check_value_duplicates

# %%
duplicates, is_negative_one_duplicate = find_column_duplicates(log._actions, check_value=-1)
print("Duplicate values by column:", duplicates)
print("Is -1 a duplicate in each column:", is_negative_one_duplicate)
    


# %%
duplicates

# %%
print(duplicates)

# %%
print(log._actions.shape)

# %%
# Sample and plot final states
s0 = one_hot(torch.zeros(10**4).long(), env.state_dim).float()
s = model.sample_states(s0, return_log=False)
# Implement your plot function or use another way to visualize the results
# plot(s, env, matrix_size)


