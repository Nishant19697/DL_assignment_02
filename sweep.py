import wandb
from train import train_and_eval
import argparse

def train_wrapper():
    wandb.init()
    config = wandb.config

    # Name the run based on important config values
    run_name = f"LR{config.learning_rate}_BS{config.batch_size}_WD{config.weight_decay}_DO{config.dropout_prob}_{config.activation}"
    wandb.run.name = run_name

    args = argparse.Namespace(
        num_layers=config.num_layers,
        hidden_size=config.hidden_size,
        activation=config.activation,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        dropout_prob=config.dropout_prob,
        batch_norm=config.batch_norm,
        weight_init=config.weight_init,
        optimizer=config.optimizer,
        epochs=config.epochs
    )

    train_and_eval(args, logging=True)

sweep_configuration = {
    'method': 'random',
    'name': 'CNet Optim Sweep',
    'metric': {'goal': 'maximize', 'name': 'val_accuracy'},
    'parameters': {
        'num_layers': {'values': [5]},
        'hidden_size': {'values': [512, 1024]},
        'activation': {'values': ['GELU', 'ReLU', 'SiLU']},
        'batch_size': {'values': [64, 128, 256]},
        'learning_rate': {'values': [0.001, 0.0005, 0.0001]},
        'weight_decay': {'values': [0, 0.005]},
        'dropout_prob': {'values': [0.3, 0.5]},  
        'batch_norm': {'values': [True, False]},
        'weight_init': {'values': ['xavier', 'kaiming', 'default']},
        'optimizer': {'values': ['adam']},
        'epochs': {'values': [10, 15, 20]}
    }
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="DL_assignment_02")
    wandb.agent(sweep_id, function=train_wrapper)
