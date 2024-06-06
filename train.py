import torch
from torch.nn.functional import one_hot
from torch.optim import Adam
from tqdm import tqdm
from preconditioner import PreconditionerEnv
from policy import ForwardPolicy, BackwardPolicy
from gflownet.gflownet import GFlowNet
from gflownet.utils import trajectory_balance_loss, market_matrix_to_sparse_tensor

def train(matrix_path: str, batch_size: int, num_epochs: int, lr: float):
    # Load the initial matrix from a file
    initial_matrix = market_matrix_to_sparse_tensor(matrix_path)
    matrix_size = initial_matrix.size(0)
    
    # Initialize the environment and policies
    env = PreconditionerEnv(matrix_size=matrix_size, initial_matrix=initial_matrix)
    node_features = 1  # Assuming each node has a single feature, can be adjusted
    hidden_dim = 32
    forward_policy = ForwardPolicy(node_features=node_features, hidden_dim=hidden_dim, num_actions=env.num_actions)
    backward_policy = BackwardPolicy(matrix_size=matrix_size, num_actions=env.num_actions)
    
    # Initialize the GFlowNet model
    model = GFlowNet(forward_policy, backward_policy, env)
    opt = Adam(model.parameters(), lr=lr)
    
    for epoch in (p := tqdm(range(num_epochs))):
        # Initialize the starting states
        s0 = one_hot(torch.zeros(batch_size).long(), env.state_dim).float()
        # Sample final states and log information
        s, log = model.sample_states(s0, return_log=True)
        
        # Calculate the trajectory balance loss
        loss = trajectory_balance_loss(log.total_flow,
                                       log.rewards,
                                       log.fwd_probs,
                                       log.back_probs)
        
        # Backpropagation and optimization step
        loss.backward()
        opt.step()
        opt.zero_grad()
        
        if epoch % 10 == 0:
            p.set_description(f"Epoch {epoch} Loss: {loss.item():.3f}")
    
    # Sample and plot final states
    s0 = one_hot(torch.zeros(10**4).long(), env.state_dim).float()
    s = model.sample_states(s0, return_log=False)
    # Implement your plot function or use another way to visualize the results
    # plot(s, env, matrix_size)

# Example usage
matrix_path = '../hangGlider_3/hangGlider_3.mtx'  # Update this with your file path
batch_size = 32
num_epochs = 1000
lr = 0.001

train(matrix_path, batch_size, num_epochs, lr)
