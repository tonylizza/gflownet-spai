import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from policy import ForwardPolicy, BackwardPolicy
from gflownet.gflownet import GFlowNet
from gflownet.dataset import MatrixDataModule

def main():
    # Initialize the data module
    matrix_dir = 'data/small_ILU'  # Replace with your actual directory
    data_module = MatrixDataModule(matrix_directory=matrix_dir, batch_size=1)
                                   
    node_features = -1
    input_dim = 1
    hidden_dim = 4
    max_num_actions = 500
    back_action_size = -1
    #Initialize the model
    forward_policy = ForwardPolicy(node_features=node_features, hidden_dim=hidden_dim, max_num_actions=max_num_actions)
    backward_policy = BackwardPolicy(input_dim=input_dim, hidden_dim=hidden_dim, max_num_actions=max_num_actions)

    logger = TensorBoardLogger("tb_logs", name="gflownet")

    model = GFlowNet(forward_policy=forward_policy, backward_policy=backward_policy, no_sampling_batch=2, lr=0.00002)

    # Initialize the PyTorch Lightning Trainer
    trainer = Trainer(max_epochs=100, logger=logger)  # You can modify trainer arguments as needed

    # Train the model
    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()