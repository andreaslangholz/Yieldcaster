import optuna
from src.utils import makeDataset, rmse, get_data
from src.models import neuralNet, kfold_cv
import torch.nn as nn
import torch
import torch.optim as optim

# Get data
df_train, df_test = get_data('wheat_winter')

# Set up CUDA
device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'
print('Device: ', device)

# Make datasets
dataset_train = makeDataset(df_train, 'yield')
dataset_test = makeDataset(df_test, 'yield')

# Static parameters
target = 'yield'
batch_size = 1024
EPOCHS = 5
K = 5
criterion = nn.MSELoss()

# Set up Study


def objective(trial, inp_features, dataset_train, criterion, epochs):
    params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
        'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
        'n_unit': trial.suggest_int("n_unit", 4, 20)
    }
    model = neuralNet(inp_features, params['n_unit'])

    optm = getattr(optim, params['optimizer'])(
        model.parameters(), lr=params['learning_rate'])

    kfold_performance = kfold_cv(
        dataset_train, model, optm, criterion, epochs=epochs)

    avg_loss = kfold_performance['avg']['test_rmse'].reset_index()[
        'test_rmse'][epochs - 1]
    return avg_loss

# Input static parameters
def obj(trial):
    return objective(trial=trial,
                     inp_features=df_train.shape[1] - 1,
                     dataset_train=dataset_train,
                     criterion=criterion,
                     epochs=EPOCHS)

# Run search and print
study_path = "C:\\Users\\Andreas Langholz\\Yieldcaster\\data\\outputs\\param1"
study_name = 'yieldcast'  # Unique identifier of the study.
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(),
                            study_name=study_name, storage=study_path)

output = study.optimize(obj, n_trials=30)

print(output)
