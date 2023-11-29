import json
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from pandas import DataFrame as df
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from my_plot_utils import raster_plot_multi_color
from my_spike_modules import BackgroundOscillationArgs, InhibitionArgs, NoiseArgs, LogFiringRateCalculationMode
from my_utils import set_seed, reorder_dataset_by_targets, FlattenTransform, ToBinaryTransform
from train_test_loop import ModelConfig, EncoderConfig, STDPConfig, OutputCellConfig, TrainConfig, train_model, \
    TestConfig, test_model

# Experiment name
experiment_name = "adaptive_simpler_non_binary_strong_inhibition"

# Set seed
seed = 9665
set_seed(seed)

# Data config
batch_size = 10
data_path = '/tmp/data/mnist'

# Data transforms
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,)),
    # ToBinaryTransform(0.5),
    FlattenTransform()
])

# Load data
mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
_, width, height = mnist_train.data.shape

# Reorder by targets
mnist_train.data, mnist_train.targets = reorder_dataset_by_targets(mnist_train.data, mnist_train.targets)
mnist_test.data, mnist_test.targets = reorder_dataset_by_targets(mnist_test.data, mnist_test.targets)

# Reduce to subset
mnist_train.data = mnist_train.data[:10000]
mnist_train.targets = mnist_train.targets[:10000]
mnist_test.data = mnist_test.data[:100]
mnist_test.targets = mnist_test.targets[:100]

# Create data loaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

distinct_targets = mnist_train.targets.unique().cpu().numpy()

# Model config
single_metric_per_batch = True

binary_input_variable_cnt = width * height
input_neuron_count = binary_input_variable_cnt * 2
output_neuron_count = 100
data_count = mnist_train.data.shape[0]

input_osc_args = BackgroundOscillationArgs(1, 20, -torch.pi / 2)
output_osc_args = BackgroundOscillationArgs(50, 20, -torch.pi / 2)

inhibition_args = InhibitionArgs(2000, 50, 2e-3)
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
        inactive_rate=0,
        background_oscillation_args=input_osc_args
    ),
    stdp_config=STDPConfig(
        base_mu=2e-1,
        base_mu_bias=2e-1,
        c=1.,
        time_batch_size=10,
        adaptive=True,
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
os.makedirs(f'./results/mnist/{experiment_name}', exist_ok=True)

# save base config
with open(f'./results/mnist/{experiment_name}/{experiment_name}_base_config.txt', 'w') as f:
    f.write(str(model_config))

train_config = TrainConfig(
    num_epochs=1,
    distinct_target_count=distinct_targets.shape[0],
    print_interval=10,

    model_config=model_config,

    single_metric_per_batch=single_metric_per_batch,
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

# Subset of test spikes
test_plot_pattern_count = 10
test_plot_timestep_subset = test_plot_pattern_count * 50

total_input_spikes = total_input_spikes[:test_plot_timestep_subset]
total_output_spikes = total_output_spikes[:test_plot_timestep_subset]
total_time_ranges = [[time_range for time_range in time_ranges_per_pattern
                      if time_range[1] <= test_plot_timestep_subset + 1]
                     for time_ranges_per_pattern in total_time_ranges]

inhibition_tracker.plot(subset_steps=test_plot_timestep_subset,
                        save_path=f'./results/mnist/{experiment_name}/{experiment_name}_{seed}_inhibition.png')

plt.plot(train_time_steps, train_results.cross_entropy_hist, label='Crossentropy')
plt.plot(train_time_steps, train_results.cross_entropy_paper_hist, label='Paper Crossentropy')
plt.title('Training')
plt.xlabel('Time [s]')
plt.ylabel('Normalized Conditional Crossentropy')
plt.ylim([0, 1])
plt.legend()
plt.savefig(f'./results/mnist/{experiment_name}/{experiment_name}_{seed}_cross_entropy.png')
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
plt.title('Training')
plt.xlabel('Time [s]')
plt.ylabel('Input log likelihood')
plt.savefig(f'./results/mnist/{experiment_name}/{experiment_name}_{seed}_input_log_likelihood.png')
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
rate_tracker.plot_relative_firing_rates(plt.gca(), colors=neuron_colors,
                                        subset_steps=test_plot_timestep_subset,
                                        legend=False)

plt.tight_layout()
plt.savefig(f'./results/mnist/{experiment_name}/{experiment_name}_{seed}_spike_history.png')
plt.show()

learning_rates_tracker.plot(
    save_path=f'./results/mnist/{experiment_name}/{experiment_name}_{seed}_learning_rates.png',
    legend=False)

# visualize bias convergence
weight_tracker.plot_bias_convergence(
    target_biases=[np.log(1. / output_neuron_count) for _ in range(output_neuron_count)],
    colors=neuron_colors, exp=False,
    save_path=f'./results/mnist/{experiment_name}/{experiment_name}_{seed}_bias_convergence.png',
    legend=False)

# visualize normalized exponential of weights in appropriate grid (10x10 for 100 output neurons)
grid_width = np.ceil(np.sqrt(output_neuron_count))
grid_height = np.ceil(output_neuron_count / grid_width)

weight_tracker.plot_final_weight_visualization((grid_width, grid_height), (width, height),
                                               save_path=f'./results/mnist/{experiment_name}/{experiment_name}_{seed}_weight_visualization.png')

test_metrics = {
    'accuracy': test_results.accuracy,
    'rate_accuracy': test_results.rate_accuracy,
    'miss_rate': test_results.miss_rate,
    'cross_entropy': test_results.cross_entropy,
    'cross_entropy_paper': test_results.cross_entropy_paper,
    'avg_log_likelihood': test_results.average_input_log_likelihood
}

with open(f'./results/mnist/{experiment_name}/{experiment_name}_test_metrics.json', 'w') as f:
    json.dump(test_metrics, f, indent=4)

with open(f'./results/mnist/{experiment_name}/{experiment_name}_train_timing_info.txt', 'w') as f:
    f.write(train_results.timing_info)

with open(f'./results/mnist/{experiment_name}/{experiment_name}_test_timing_info.txt', 'w') as f:
    f.write(test_results.timing_info)
