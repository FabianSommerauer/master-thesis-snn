import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from binary_pattern_dataset import BinaryPatternDataset
from my_spike_modules import InhibitionArgs, NoiseArgs, LogFiringRateCalculationMode, BackgroundOscillationArgs
from my_utils import set_seed
from train_test_loop import ModelConfig, EncoderConfig, STDPConfig, OutputCellConfig, TrainConfig, TestConfig, \
    evaluate_config

# Set seed
seed = 2342
set_seed(seed)

# Data config
batch_size = 1
num_patterns = 10
num_repeats_train = 100
num_repeats_test = 10
pattern_length = 300
pattern_sparsity = 0.5


def init_binary_pattern_dataset(seed=None):
    # Load data
    binary_train = BinaryPatternDataset(num_patterns, num_repeats_train, pattern_length, pattern_sparsity, seed=seed)
    binary_test = BinaryPatternDataset(num_patterns, num_repeats_test, pattern_length, pattern_sparsity, seed=seed)

    train_loader = DataLoader(binary_train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(binary_test, batch_size=batch_size, shuffle=False)

    # distinct_targets = binary_train.pattern_ids.unique().cpu().numpy()

    return train_loader, test_loader


# Model config
binary_input_variable_cnt = pattern_length
input_neuron_count = binary_input_variable_cnt * 2
output_neuron_count = num_patterns

input_osc_args = None  # BackgroundOscillationArgs(1, 20, -torch.pi / 2)
output_osc_args = BackgroundOscillationArgs(50, 20, -torch.pi / 2)

inhibition_args = InhibitionArgs(2000, 100, 5e-3)  # 1000, 0, 5e-3 (classic); 2000, 100, 5e-3 (adaptive)
noise_args = NoiseArgs(0, 5e-3, 50)

model_config = ModelConfig(
    dt=1e-3,
    input_neuron_count=input_neuron_count,
    output_neuron_count=output_neuron_count,
    sigma=5e-3,

    encoder_config=EncoderConfig(
        presentation_duration=4e-2,
        delay=1e-2,
        active_rate=40,
        inactive_rate=5,
        background_oscillation_args=input_osc_args
    ),
    stdp_config=STDPConfig(
        base_mu=2e-1,  # 5e-1 (classic); 2e-1 (adaptive)
        base_mu_bias=1e-1,  # 5e-1 (classic); 1e-1 (adaptive)
        c=1.,
        time_batch_size=5,
        adaptive=True,
    ),
    output_cell_config=OutputCellConfig(
        inhibition_args=inhibition_args,
        noise_args=noise_args,
        log_firing_rate_calc_mode=LogFiringRateCalculationMode.ExpectedInputCorrected,
        background_oscillation_args=output_osc_args,
    ),

    weight_init=0,  # 0 (classic); 2 (adaptive)
    bias_init=0  # -2 (classic); 2 (adaptive)
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
    print_results=True,
)

repeats = 5
seeds = [random.randint(0, 10000) for _ in range(repeats)]

experiment_name = 'adaptive_test'


def print_eval_results(res):
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


train_loader, test_loader = init_binary_pattern_dataset()
eval_results = evaluate_config(train_config, test_config, init_binary_pattern_dataset, seeds=seeds)
print_eval_results(eval_results)

# save results
eval_results['confusion_matrix'] = [mat.tolist() for mat in eval_results['confusion_matrix']]
with open(f'./results/{experiment_name}_{seed}.json', 'w') as f:
    json.dump(eval_results, f, indent=4)
