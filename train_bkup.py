import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from policy import ForwardPolicy, BackwardPolicy
from gflownet.gflownet import GFlowNet
from gflownet.dataset import MatrixDataModule
import time
import os

def run_experiment(hyperparams):
    num_workers = os.cpu_count()
    start_time = time.time()
    matrix_dir = 'data/small_ILU'
    data_module = MatrixDataModule(matrix_directory=matrix_dir, num_workers=num_workers, batch_size=1)

    forward_policy = ForwardPolicy(
        node_features=hyperparams['node_features'],
        hidden_dim=hyperparams['hidden_dim'],
        max_num_actions=hyperparams['max_num_actions']
    )
    backward_policy = BackwardPolicy(
        input_dim=hyperparams['input_dim'],
        hidden_dim=hyperparams['hidden_dim'],
        max_num_actions=hyperparams['max_num_actions']
    )

    model = GFlowNet(
        forward_policy=forward_policy,
        backward_policy=backward_policy,
        no_sampling_batch=hyperparams['no_sampling_batch'],
        lr=hyperparams['lr'],
        schedule_patience=hyperparams['schedule_patience']
    )

    logger = TensorBoardLogger("tb_logs", name=f"gflownet_lr_{hyperparams['lr']}_epochs_{hyperparams['number_epoch']}_sampling_{hyperparams['no_sampling_batch']}_patience_{hyperparams['schedule_patience']}")

    callbacks = [
        EarlyStopping(monitor="train_loss", mode="min", patience=10),
        ModelCheckpoint(monitor="train_loss", save_top_k=3, mode="min")
    ]

    trainer = pl.Trainer(max_epochs=hyperparams['number_epoch'], logger=logger, callbacks=callbacks)
    trainer.fit(model, data_module)
    training_time = time.time() - start_time
    print(f"Elapsed Training Time: {training_time}")

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, required=True, help="Learning rate for the experiment")
    parser.add_argument('--epochs', type=int, required=True, help="Number of training epochs")
    parser.add_argument('--no_sampling_batch', type=int, required=True, help="Number of sampling batches")
    parser.add_argument('--patience', type=int, required=True, help="Schedule patience for early stopping")

    args = parser.parse_args()

    # Define hyperparameters based on parsed arguments
    hyperparams = {
        'lr': args.lr,
        'number_epoch': args.epochs,
        'no_sampling_batch': args.no_sampling_batch,
        'hidden_dim': 2,
        'node_features': -1,
        'input_dim': 1,
        'max_num_actions': 180000,
        'schedule_patience': args.patience
    }

    run_experiment(hyperparams)
