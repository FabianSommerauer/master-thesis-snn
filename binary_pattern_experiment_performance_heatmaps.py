import dataclasses
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from binary_pattern_dataset import BinaryPatternDataset, BinaryPatternDataConfig
from my_spike_modules import InhibitionArgs, NoiseArgs, LogFiringRateCalculationMode, InputBackgroundOscillationArgs, \
    BackgroundOscillationArgs
from my_utils import set_seed
from train_test_loop import ModelConfig, EncoderConfig, STDPConfig, OutputCellConfig, TrainConfig, TestConfig, \
    evaluate_config, STDPAdaptiveConfig

# Set seed
seed = 93259
set_seed(seed)

# Data config
batch_size = 1
num_patterns = 10
num_repeats_train = 100
num_repeats_test = 10
pattern_length = 100
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

input_osc_args = InputBackgroundOscillationArgs(0.5, 20, -torch.pi / 2, 0.5)
output_osc_args = BackgroundOscillationArgs(50, 20, -torch.pi / 2)

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


def print_eval_results(experiment_name, values, res):
    print("---------------------------------")
    print(f"Experiment: {experiment_name}; Value: {values}")
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


def set_active_firing_rate(model_config: ModelConfig, data_config: BinaryPatternDataConfig,
                           value: float):
    model_config.encoder_config.active_rate = value


def set_inactive_firing_rate(model_config: ModelConfig, data_config: BinaryPatternDataConfig,
                             value: float):
    model_config.encoder_config.inactive_rate = value


def set_num_patterns(model_config: ModelConfig, data_config: BinaryPatternDataConfig,
                     value: int):
    data_config.num_patterns = value


def set_pattern_length(model_config: ModelConfig, data_config: BinaryPatternDataConfig,
                       value: int):
    data_config.pattern_length = value


repeats = 1
seeds = [random.randint(0, 10000) for _ in range(repeats)]

experiment_name = 'active_vs_inactive'
param_name_1 = 'Active Firing rate'
set_param_func_1 = set_active_firing_rate
param_values_1 = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
set_param_func_2 = set_inactive_firing_rate
param_name_2 = 'Inactive Firing rate'
param_values_2 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# create folder for experiment
os.makedirs(f'./results/config_eval_heatmap/{experiment_name}', exist_ok=True)

# save base config
with open(f'./results/config_eval_heatmap/{experiment_name}/{experiment_name}_base_config.json', 'w') as f:
    json.dump(dataclasses.asdict(model_config), f, indent=4)
with open(f'./results/config_eval_heatmap/{experiment_name}/{experiment_name}_data_config.json', 'w') as f:
    json.dump(dataclasses.asdict(data_config), f, indent=4)

# Metrics
val_1_count = len(param_values_1)
val_2_count = len(param_values_2)

accuracies = np.zeros((val_1_count, val_2_count), dtype=float)
rate_accuracies = np.zeros((val_1_count, val_2_count), dtype=float)
losses = np.zeros((val_1_count, val_2_count), dtype=float)
paper_losses = np.zeros((val_1_count, val_2_count), dtype=float)

# Run experiment
for i, val_1 in enumerate(param_values_1):
    for j, val_2 in enumerate(param_values_2):
        set_param_func_1(model_config, data_config, val_1)
        set_param_func_2(model_config, data_config, val_2)

        train_config.model_config = model_config
        test_config.model_config = model_config

        eval_results = evaluate_config(train_config, test_config, init_binary_pattern_dataset, seeds=seeds)
        print_eval_results(experiment_name, [val_1, val_2], eval_results)

        # add results to metrics
        accuracies[i, j] = eval_results['accuracy']
        rate_accuracies[i, j] = eval_results['rate_accuracy']
        losses[i, j] = eval_results['loss']
        paper_losses[i, j] = eval_results['loss_paper']

sns.heatmap(accuracies, xticklabels=param_values_1, yticklabels=param_values_2)
plt.title('Accuracy')
plt.xlabel(param_name_1)
plt.ylabel(param_name_2)
plt.tight_layout()
plt.savefig(f'./results/config_eval_heatmap/{experiment_name}/{experiment_name}_{seed}_accuracy.png')
plt.show()

sns.heatmap(rate_accuracies, xticklabels=param_values_1, yticklabels=param_values_2)
plt.title('Rate Accuracy')
plt.xlabel(param_name_1)
plt.ylabel(param_name_2)
plt.tight_layout()
plt.savefig(f'./results/config_eval_heatmap/{experiment_name}/{experiment_name}_{seed}_rate_accuracy.png')
plt.show()

sns.heatmap(losses, xticklabels=param_values_1, yticklabels=param_values_2)
plt.title('Normalized conditional entropy')
plt.xlabel(param_name_1)
plt.ylabel(param_name_2)
plt.tight_layout()
plt.savefig(f'./results/config_eval_heatmap/{experiment_name}/{experiment_name}_{seed}_loss.png')
plt.show()

sns.heatmap(paper_losses, xticklabels=param_values_1, yticklabels=param_values_2)
plt.title('Normalized conditional entropy paper')
plt.xlabel(param_name_1)
plt.ylabel(param_name_2)
plt.tight_layout()
plt.savefig(f'./results/config_eval_heatmap/{experiment_name}/{experiment_name}_{seed}_loss_paper.png')
plt.show()
