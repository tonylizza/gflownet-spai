import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from policy import ForwardPolicy, BackwardPolicy
from gflownet.gflownet import GFlowNet
from gflownet.dataset import MatrixDataModule
import itertools
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from gflownet.gflownet import GFlowNet
from gflownet.dataset import MatrixDataModule


def run_experiment(hyperparams):
    matrix_dir = 'data/large_ILU'
    data_module = MatrixDataModule(matrix_directory=matrix_dir, batch_size=1)

    forward_policy = ForwardPolicy(node_features=hyperparams['node_features'], hidden_dim=hyperparams['hidden_dim'], max_num_actions=hyperparams['max_num_actions'])
    backward_policy = BackwardPolicy(input_dim=hyperparams['input_dim'], hidden_dim=hyperparams['hidden_dim'], max_num_actions=hyperparams['max_num_actions'])

    model = GFlowNet(forward_policy=forward_policy, backward_policy=backward_policy, no_sampling_batch=hyperparams['no_sampling_batch'], lr=hyperparams['lr'], schedule_patience=hyperparams['schedule_patience'])

    logger = TensorBoardLogger("tb_logs", name=f"gflownet_lr_{hyperparams['lr']}_epochs_{hyperparams['number_epoch']}_sampling_{hyperparams['no_sampling_batch']}_patience_{hyperparams['schedule_patience']}")

    callbacks = [
        EarlyStopping(monitor="train_loss", mode="min", patience=10),
        ModelCheckpoint(monitor="train_loss", save_top_k=3, mode="min")
    ]

    trainer = pl.Trainer(max_epochs=hyperparams['number_epoch'], logger=logger, callbacks=callbacks)
    trainer.fit(model, data_module)

if __name__ == '__main__':
    # Define hyperparameters space
    learning_rates = [2e-4, 7e-5, 2e-5]
    number_epochs = [2, 3] #Change to 50, 100 after testing
    no_sampling_batches = [2, 8, 16] #Change to 4, 8, 16 after testing
    schedule_patience = [5, 10] 

    # Create hyperparameter combinations
    hyperparams_combinations = list(itertools.product(learning_rates, number_epochs, no_sampling_batches, schedule_patience))

    # Run experiments for each combination
    for lr, number_epoch, no_sampling_batch, patience in hyperparams_combinations:
        hyperparams = {
            'lr': lr,
            'number_epoch': number_epoch,
            'no_sampling_batch': no_sampling_batch,
            'hidden_dim': 4,
            'node_features': -1,
            'input_dim': 1,
            'max_num_actions': 18000000,
            'schedule_patience': patience
        }
        run_experiment(hyperparams)