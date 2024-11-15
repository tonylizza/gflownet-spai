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
import time
import os



if __name__ == '__main__':
    # Define hyperparameters space
    learning_rates = 2e-6
    number_epochs = 100 #Change to 50, 100 after testing
    no_sampling_batches = 1 #Change to 4, 8, 16 after testing
    schedule_patience = 5 


    # Run experiments for each combination

    hyperparams = {
        'lr': learning_rates,
        'number_epoch': number_epochs,
        'no_sampling_batch': no_sampling_batches,
        'hidden_dim': 2,
        'node_features': -1,
        'input_dim': 1,
        'max_num_actions': 180000,
        'schedule_patience': schedule_patience
    }

    num_workers = os.cpu_count()
    start_time = time.time()
    matrix_dir = 'data/medium_ILU'
    data_module = MatrixDataModule(matrix_directory=matrix_dir, num_workers=num_workers, batch_size=1)

    forward_policy = ForwardPolicy(node_features=hyperparams['node_features'], hidden_dim=hyperparams['hidden_dim'], max_num_actions=hyperparams['max_num_actions'])
    backward_policy = BackwardPolicy(input_dim=hyperparams['input_dim'], hidden_dim=hyperparams['hidden_dim'], max_num_actions=hyperparams['max_num_actions'])

    model = GFlowNet(forward_policy=forward_policy, backward_policy=backward_policy, no_sampling_batch=hyperparams['no_sampling_batch'], lr=hyperparams['lr'], schedule_patience=hyperparams['schedule_patience'])

    logger = TensorBoardLogger("tb_logs_final", name=f"gflownet_lr_{hyperparams['lr']}_epochs_{hyperparams['number_epoch']}_sampling_{hyperparams['no_sampling_batch']}_patience_{hyperparams['schedule_patience']}")

    callbacks = [
        EarlyStopping(monitor="train_loss", mode="min", patience=10),
        ModelCheckpoint(monitor="train_loss", save_top_k=3, mode="min")
    ]

    trainer = pl.Trainer(max_epochs=hyperparams['number_epoch'], logger=logger, callbacks=callbacks)
    trainer.fit(model, data_module)
    training_time = time.time() - start_time
    print(f"Elapsed Training Time: {training_time}")