import dataclasses
import json
import os
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from binary_pattern_dataset import BinaryPatternDataset
from my_spike_modules import InhibitionArgs, NoiseArgs, LogFiringRateCalculationMode
from my_utils import set_seed
from train_test_loop import ModelConfig, EncoderConfig, STDPConfig, OutputCellConfig, TrainConfig, TestConfig, \
    evaluate_config, STDPAdaptiveConfig, STDPClassicConfig


@dataclass
class BinaryPatternDataConfig:
    batch_size: int
    num_patterns: int
    num_repeats_train: int
    num_repeats_test: int
    pattern_length: int
    pattern_sparsity: float


# Set seed
seed = 4341
set_seed(seed)

# Data config
batch_size = 1
num_patterns = 10
num_repeats_train = 100
num_repeats_test = 10
pattern_length = 300
pattern_sparsity = 0.5

data_config = BinaryPatternDataConfig(batch_size, num_patterns, num_repeats_train,
                                      num_repeats_test, pattern_length, pattern_sparsity)


def init_binary_pattern_dataset(seed=None):
    # Load data
    binary_train = BinaryPatternDataset(data_config.num_patterns, data_config.num_repeats_train,
                                        data_config.pattern_length, data_config.pattern_sparsity, seed=seed)
    binary_test = BinaryPatternDataset(data_config.num_patterns, data_config.num_repeats_test,
                                       data_config.pattern_length, data_config.pattern_sparsity, seed=seed)

    train_loader = DataLoader(binary_train, batch_size=data_config.batch_size, shuffle=False)
    test_loader = DataLoader(binary_test, batch_size=data_config.batch_size, shuffle=False)

    return train_loader, test_loader


# Model config
binary_input_variable_cnt = pattern_length
input_neuron_count = binary_input_variable_cnt * 2
output_neuron_count = num_patterns

input_osc_args = None  # BackgroundOscillationArgs(1, 20, -torch.pi / 2)
output_osc_args = None  # BackgroundOscillationArgs(50, 20, -torch.pi / 2)

inhibition_args = InhibitionArgs(2000, 100, 5e-3)  # 1000, 0, 2e-3 (weak); 2000, 100, 5e-3 (strong)
noise_args = NoiseArgs(0, 5e-3, 50)

model_config = ModelConfig(
    dt=1e-3,
    input_neuron_count=input_neuron_count,
    output_neuron_count=output_neuron_count,
    sigma=5e-3,

    encoder_config=EncoderConfig(
        presentation_duration=4e-2,
        delay=1e-2,
        active_rate=30,
        inactive_rate=5,
        background_oscillation_args=input_osc_args
    ),
    stdp_config=STDPConfig(
        c=1.,
        time_batch_size=5,
        method=STDPAdaptiveConfig(base_mu=5e-1, base_mu_bias=5e-1)
        # method=STDPClassicConfig(base_mu=1., base_mu_bias=1.)
    ),
    output_cell_config=OutputCellConfig(
        inhibition_args=inhibition_args,
        noise_args=noise_args,
        log_firing_rate_calc_mode=LogFiringRateCalculationMode.ExpectedInputCorrected,
        background_oscillation_args=output_osc_args,
    ),

    weight_init=0,
    bias_init=0
)

train_config = TrainConfig(
    num_epochs=1,
    distinct_target_count=num_patterns,
    print_interval=None,

    model_config=model_config,
)

test_config = TestConfig(
    distinct_target_count=num_patterns,
    model_config=model_config,
    print_results=False,
)


def print_eval_results(experiment_name, value, res):
    print("---------------------------------")
    print(f"Experiment: {experiment_name}; Value: {value}")
    print("---------------------------------")
    print(f"Average accuracy: {np.mean(res['accuracy'])}")
    print(f"Std accuracy: {np.std(res['accuracy'])}")
    print(f"Average rate accuracy: {np.mean(res['rate_accuracy'])}")
    print(f"Std rate accuracy: {np.std(res['rate_accuracy'])}")
    print(f"Average miss rate: {np.mean(res['miss_rate'])}")
    print(f"Std miss rate: {np.std(res['miss_rate'])}")
    print(f"Average loss: {np.mean(res['loss'])}")
    print(f"Std loss: {np.std(res['loss'])}")
    print(f"Average loss paper: {np.mean(res['loss_paper'])}")
    print(f"Std loss paper: {np.std(res['loss_paper'])}")
    print(f"Average input log likelihood: {np.mean(res['input_log_likelihood'])}")
    print(f"Std input log likelihood: {np.std(res['input_log_likelihood'])}")
    print("---------------------------------\n")


def set_inhibition_rest(model_config: ModelConfig, data_config: BinaryPatternDataConfig, value: float):
    model_config.output_cell_config.inhibition_args.inhibition_rest = value


def set_adaptive(model_config: ModelConfig, data_config: BinaryPatternDataConfig, value: bool):
    if value:
        model_config.stdp_config.method = STDPAdaptiveConfig(base_mu=5e-1, base_mu_bias=5e-1)
    else:
        model_config.stdp_config.method = STDPClassicConfig(base_mu=1., base_mu_bias=1.)


def set_background_oscillation(model_config: ModelConfig, data_config: BinaryPatternDataConfig, value: str):
    if value == 'None':
        model_config.encoder_config.background_oscillation_args = None
        model_config.output_cell_config.background_oscillation_args = None
    elif value == 'Input Only':
        model_config.encoder_config.background_oscillation_args = input_osc_args
        model_config.output_cell_config.background_oscillation_args = None
    elif value == 'Output Only':
        model_config.encoder_config.background_oscillation_args = None
        model_config.output_cell_config.background_oscillation_args = output_osc_args
    elif value == 'Both':
        model_config.encoder_config.background_oscillation_args = input_osc_args
        model_config.output_cell_config.background_oscillation_args = output_osc_args
    else:
        raise ValueError(f'Unknown value: {value}')


repeats = 5
seeds = [random.randint(0, 10000) for _ in range(repeats)]

# experiment_name = 'inhibition_rest'
# param_name = 'inhibition_rest'
# set_param_func = set_inhibition_rest
# param_values = [0, 25, 50, 75, 100, 125, 150, 175, 200]
# values_categorical = False

experiment_name = 'adaptive_strong_inhibition'
param_name = 'adaptive'
set_param_func = set_adaptive
param_values = [True, False]
values_categorical = True

# experiment_name = 'background_oscillation'
# param_name = 'Oscillation Type'
# set_param_func = set_background_oscillation
# param_values = ['None', 'Input Only', 'Output Only', 'Both']
# values_categorical = True

# create folder for experiment
os.makedirs(f'./results/config_eval/{experiment_name}', exist_ok=True)

# save base config
with open(f'./results/config_eval/{experiment_name}/{experiment_name}_base_config.json', 'w') as f:
    json.dump(dataclasses.asdict(model_config), f, indent=4)
with open(f'./results/config_eval/{experiment_name}/{experiment_name}_data_config.json', 'w') as f:
    json.dump(dataclasses.asdict(data_config), f, indent=4)

# Train loss
avg_train_loss = None
avg_train_loss_paper = None
train_losses = []
train_losses_paper = []

# Dataframe list
dfs = []

# Run experiment
for val in param_values:
    set_param_func(model_config, data_config, val)
    train_config.model_config = model_config
    test_config.model_config = model_config

    eval_results = evaluate_config(train_config, test_config, init_binary_pattern_dataset, seeds=seeds)
    print_eval_results(experiment_name, val, eval_results)

    # update train loss
    train_losses.append(eval_results['avg_train_loss'])
    train_losses_paper.append(eval_results['avg_train_loss_paper'])
    if avg_train_loss is None:
        avg_train_loss = eval_results['avg_train_loss']
    else:
        avg_train_loss += eval_results['avg_train_loss']

    if avg_train_loss_paper is None:
        avg_train_loss_paper = eval_results['avg_train_loss_paper']
    else:
        avg_train_loss_paper += eval_results['avg_train_loss_paper']

    # # save results
    # eval_results['confusion_matrix'] = [mat.tolist() for mat in eval_results['confusion_matrix']]
    # with open(f'./results/config_eval/{experiment_name}/{experiment_name}_{val}_{seed}.json', 'w') as f:
    #     json.dump(eval_results, f, indent=4)

    # add results to dataframe
    new_df = pd.DataFrame({
        'experiment': [experiment_name] * repeats,
        'seed': [seed] * repeats,
        'value': [val] * repeats,
        'accuracy': eval_results['accuracy'],
        'rate_accuracy': eval_results['rate_accuracy'],
        'miss_rate': eval_results['miss_rate'],
        'loss': eval_results['loss'],
        'loss_paper': eval_results['loss_paper'],
        'input_log_likelihood': eval_results['input_log_likelihood'],
        'confusion_matrix_str': [str(mat) for mat in eval_results['confusion_matrix']]
    })
    dfs.append(new_df)

# average train loss
avg_train_loss /= len(param_values)
avg_train_loss_paper /= len(param_values)

# save average train loss
with open(f'./results/config_eval/{experiment_name}/{experiment_name}_{seed}_avg_train_loss.json', 'w') as f:
    json.dump(avg_train_loss.tolist(), f, indent=4)
with open(f'./results/config_eval/{experiment_name}/{experiment_name}_{seed}_avg_train_loss_paper.json', 'w') as f:
    json.dump(avg_train_loss_paper.tolist(), f, indent=4)

df = pd.concat(dfs)
if values_categorical:
    df['value'] = pd.Categorical(df['value'], categories=param_values, ordered=True)

# Save dataframe
df.to_csv(f'./results/config_eval/{experiment_name}/{experiment_name}_{seed}.csv', index=False)

# Plot results as boxplots
plt.rc('font', size=18)
boxprops = dict(linestyle='-', linewidth=2, color='k')
whiskerprops = dict(linestyle='-', linewidth=2, color='k')
capprops = dict(linestyle='-', linewidth=2, color='k')
medianprops = dict(linestyle='-', linewidth=2, color='g')

columns = ['accuracy', 'rate_accuracy', 'miss_rate', 'loss', 'loss_paper', 'input_log_likelihood']
for column in columns:
    _ = df.boxplot(column=column,
                   by='value', figsize=(8, 8),
                   boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, medianprops=medianprops)
    plt.xlabel(param_name)
    plt.ylabel(column)
    if column in ['accuracy', 'rate_accuracy', 'miss_rate', 'loss', 'loss_paper']:
        plt.ylim([0, 1])
    plt.title('')
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(f'./results/config_eval/{experiment_name}/{experiment_name}_{seed}_{column}.png')
    plt.show()

plt.rc('font', size=12)

# plot average train loss
plt.plot(avg_train_loss, label='Train loss')
plt.plot(avg_train_loss_paper, label='Train loss paper')
plt.title('Training')
plt.xlabel('Time')
plt.ylim([0, 1])
plt.tight_layout()
plt.legend()
plt.savefig(f'./results/config_eval/{experiment_name}/{experiment_name}_{seed}_avg_train_loss.png')
plt.show()

# plot each loss
for i in range(len(train_losses)):
    plt.plot(train_losses[i], label=f'{param_values[i]}')
plt.title('Training')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.ylim([0, 1])
plt.tight_layout()
plt.legend(title=f'{param_name}')
plt.savefig(f'./results/config_eval/{experiment_name}/{experiment_name}_{seed}_train_loss.png')
plt.show()

for i in range(len(train_losses_paper)):
    plt.plot(train_losses_paper[i], label=f'{param_values[i]}')
plt.title('Training')
plt.xlabel('Time')
plt.ylabel('Paper Loss')
plt.ylim([0, 1])
plt.tight_layout()
plt.legend(title=f'{param_name}')
plt.savefig(f'./results/config_eval/{experiment_name}/{experiment_name}_{seed}_train_loss_paper.png')
plt.show()
