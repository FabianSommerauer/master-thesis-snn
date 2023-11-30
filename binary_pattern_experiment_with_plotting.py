import dataclasses
import json
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from pandas import DataFrame as df
from torch.utils.data import DataLoader

from binary_pattern_dataset import BinaryPatternDataset
from my_plot_utils import raster_plot_multi_color
from my_spike_modules import InhibitionArgs, NoiseArgs, LogFiringRateCalculationMode, OutputBackgroundOscillationArgs, \
    InputBackgroundOscillationArgs
from my_utils import set_seed
from train_test_loop import ModelConfig, EncoderConfig, STDPConfig, OutputCellConfig, TrainConfig, train_model, \
    TestConfig, test_model, STDPAdaptiveConfig

# Experiment name
experiment_name = "adaptive_small_100Hz_5Hz"

# Set seed
seed = 56423
set_seed(seed)

# Data config
batch_size = 1
num_patterns = 5
num_repeats_train = 100
num_repeats_test = 4
pattern_length = 100
pattern_sparsity = 0.5

# Load data
binary_train = BinaryPatternDataset(num_patterns, num_repeats_train, pattern_length, pattern_sparsity, seed=seed)
binary_test = BinaryPatternDataset(num_patterns, num_repeats_test, pattern_length, pattern_sparsity, seed=seed)

train_loader = DataLoader(binary_train, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(binary_test, batch_size=batch_size, shuffle=False)

distinct_targets = binary_train.pattern_ids.unique().cpu().numpy()

# Model config
pat_len = binary_train.patterns.shape[1]
binary_input_variable_cnt = pat_len
input_neuron_count = binary_input_variable_cnt * 2
output_neuron_count = distinct_targets.shape[0]

input_osc_args = InputBackgroundOscillationArgs(0.5, 0.5, 20, -torch.pi / 2)
output_osc_args = OutputBackgroundOscillationArgs(50, 20, -torch.pi / 2)

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
        active_rate=100,
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

# create folder for experiment
os.makedirs(f'./results/single_run/{experiment_name}', exist_ok=True)

# save base config
with open(f'./results/single_run/{experiment_name}/{experiment_name}_model_config.json', 'w') as f:
    json.dump(dataclasses.asdict(model_config), f, indent=4)

with open(f'./results/single_run/{experiment_name}/{experiment_name}_data_config.json', 'w') as f:
    json.dump({
        'batch_size': batch_size,
        'num_patterns': num_patterns,
        'num_repeats_train': num_repeats_train,
        'num_repeats_test': num_repeats_test,
        'pattern_length': pattern_length,
        'pattern_sparsity': pattern_sparsity,
    }, f, indent=4)

train_config = TrainConfig(
    num_epochs=1,
    distinct_target_count=distinct_targets.shape[0],
    print_interval=10,

    model_config=model_config,
)

test_config = TestConfig(
    distinct_target_count=distinct_targets.shape[0],
    model_config=model_config,
    print_results=True,
)

# Train
train_results = train_model(train_config, train_loader)

trained_params = train_results.trained_params
neuron_pattern_mapping = train_results.neuron_pattern_mapping

test_config.trained_params = trained_params
test_config.neuron_pattern_mapping = neuron_pattern_mapping

test_results = test_model(test_config, test_loader)

# Get spikes
total_input_spikes = test_results.total_input_spikes
total_output_spikes = test_results.total_output_spikes
total_time_ranges = test_results.total_time_ranges

# Get trackers
learning_rates_tracker = train_results.learning_rates_tracker
weight_tracker = train_results.weights_tracker

rate_tracker = test_results.rate_tracker
inhibition_tracker = test_results.inhibition_tracker

# Plot
print("Plotting results...")

train_time_steps = np.arange(1, train_results.cross_entropy_hist.shape[0] + 1)
train_time_steps = train_time_steps * model_config.dt  # convert to seconds
if train_config.single_metric_per_batch:
    train_time_steps *= 50  # ms per data point
    train_time_steps *= batch_size
    train_data_time_steps = train_time_steps
else:
    train_data_time_steps = train_time_steps[::batch_size * 50]

inhibition_tracker.plot(save_path=f'./results/single_run/{experiment_name}/{experiment_name}_{seed}_inhibition.png')

plt.plot(train_time_steps, train_results.cross_entropy_hist, label='Crossentropy')
plt.plot(train_time_steps, train_results.cross_entropy_paper_hist, label='Paper Crossentropy')
plt.title('Training')
plt.xlabel('Time [s]')
plt.ylabel('Normalized Conditional Crossentropy')
plt.ylim([0, 1])
plt.legend()
plt.savefig(f'./results/single_run/{experiment_name}/{experiment_name}_{seed}_cross_entropy.png')
plt.show()

input_log_likelihood_df = df.from_dict({
    'ts': train_data_time_steps,
    'log_l': train_results.input_log_likelihood_hist,
})

rolling_mean = input_log_likelihood_df.log_l.rolling(window=10).mean()
rolling_std = input_log_likelihood_df.log_l.rolling(window=10).std()

# plt.plot(train_data_time_steps, train_results.input_log_likelihood_hist, label='Input log likelihood')
plt.plot(train_data_time_steps, rolling_mean, label='Input log likelihood')
plt.fill_between(train_data_time_steps, rolling_mean - rolling_std, rolling_mean + rolling_std, alpha=0.2)
# plt.axhline(y=np.log(1. / num_patterns), color='r', linestyle='-', label='Maximum Avg. Input Log Likelihood')
plt.title('Training')
plt.xlabel('Time [s]')
plt.ylabel('Input log likelihood')
plt.savefig(f'./results/single_run/{experiment_name}/{experiment_name}_{seed}_input_log_likelihood.png')
# plt.legend()
plt.show()

cmap = plt.get_cmap("tab10")
group_colors = [cmap(i) for i in range(distinct_targets.shape[0])]
allowed_colors = [[idx, ] for idx in neuron_pattern_mapping]

# Plot inputs
plt.figure(figsize=(10, 7))

plt.subplot(3, 1, 1)
first_ax = plt.gca()
spikes = [total_input_spikes[:, i] for i in range(total_input_spikes.shape[1])]
# raster_plot(plt.gca(), spikes)
raster_plot_multi_color(plt.gca(), spikes, total_time_ranges, group_colors)
plt.title('Input Spikes')
plt.xlabel('Time Step [ms]')
plt.ylabel('Neuron')

# Plot output spikes using a raster plot
plt.subplot(3, 1, 2)
plt.gca().sharex(first_ax)
spikes = [total_output_spikes[:, i] for i in range(total_output_spikes.shape[1])]
raster_plot_multi_color(plt.gca(), spikes, total_time_ranges, group_colors, default_color='black',
                        allowed_colors_per_train=allowed_colors)
plt.title('Output Spikes')
plt.xlabel('Time Step [ms]')
plt.ylabel('Neuron')

plt.subplot(3, 1, 3)
plt.gca().sharex(first_ax)
neuron_colors = [group_colors[idx] for idx in train_results.neuron_pattern_mapping]
rate_tracker.plot_relative_firing_rates(plt.gca(), colors=neuron_colors)

plt.tight_layout()
plt.savefig(f'./results/single_run/{experiment_name}/{experiment_name}_{seed}_spike_history.png')
plt.show()

learning_rates_tracker.plot(
    save_path=f'./results/single_run/{experiment_name}/{experiment_name}_{seed}_learning_rates.png',
    legend=False)

# visualize bias convergence
weight_tracker.plot_bias_convergence(target_biases=[np.log(1. / output_neuron_count)
                                                    for _ in range(output_neuron_count)],
                                     colors=neuron_colors, exp=False,
                                     save_path=f'./results/single_run/{experiment_name}/{experiment_name}_{seed}_bias_convergence.png',
                                     legend=False)

# visualize normalized exponential of weights in appropriate grid (10x10 for 100 output neurons)
grid_width = np.ceil(np.sqrt(output_neuron_count))
grid_height = np.ceil(output_neuron_count / grid_width)
width = np.ceil(np.sqrt(pat_len))
height = np.ceil(pat_len / width)
weight_tracker.plot_final_weight_visualization((grid_width, grid_height), (width, height),
                                               save_path=f'./results/single_run/{experiment_name}/{experiment_name}_{seed}_weight_visualization.png')

test_metrics = {
    'accuracy': test_results.accuracy,
    'rate_accuracy': test_results.rate_accuracy,
    'miss_rate': test_results.miss_rate,
    'cross_entropy': test_results.cross_entropy,
    'cross_entropy_paper': test_results.cross_entropy_paper,
    'avg_log_likelihood': test_results.average_input_log_likelihood
}

with open(f'./results/single_run/{experiment_name}/{experiment_name}_test_metrics.json', 'w') as f:
    json.dump(test_metrics, f, indent=4)

with open(f'./results/single_run/{experiment_name}/{experiment_name}_train_timing_info.txt', 'w') as f:
    f.write(train_results.timing_info)

with open(f'./results/single_run/{experiment_name}/{experiment_name}_test_timing_info.txt', 'w') as f:
    f.write(test_results.timing_info)
