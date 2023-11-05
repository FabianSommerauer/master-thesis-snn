import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import random
import torchmetrics

import custom_stdp
from my_spike_modules import *
from my_utils import spike_in_range, get_neuron_pattern_mapping, get_predictions, get_joint_probabilities_over_time, \
    normalized_conditional_cross_entropy, normalized_conditional_cross_entropy_paper, get_cumulative_counts_over_time, \
    get_joint_probabilities_from_counts
from my_plot_utils import raster_plot, raster_plot_multi_color, raster_plot_multi_color_per_train
from my_timing_utils import Timer

torch.set_grad_enabled(False)

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Set seed (todo: test with different seeds)
seed = 45
random.seed(seed)
_ = torch.manual_seed(seed)
np.random.seed(seed)

# Data config
batch_size = 10
data_path = '/tmp/data/mnist'


class ToBinaryTransform(object):
    def __init__(self, thresh):
        self.thresh = thresh

    def __call__(self, data: Tensor):
        return (data > self.thresh).to(data.dtype)


# Data transforms
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,)),
    ToBinaryTransform(0.5)  # todo: test without this
])

# Load data
mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

distinct_targets = mnist_train.targets.unique().cpu().numpy()

# Model config
_, width, height = mnist_train.data.shape
binary_input_variable_cnt = width * height
input_neurons = binary_input_variable_cnt * 2
output_neurons = 100

dt = 0.001
sigma = 0.005

presentation_duration = 0.04
delay = 0.01

# todo: experiment with different rates (maybe different rates for train and test as well)
input_encoding_rate = 100
input_encoding_inactive_rate = 10

stdp_time_batch_size = 10

# Model setup
train_encoder = SpikePopulationGroupBatchToTimeEncoder(presentation_duration,
                                                       input_encoding_rate, input_encoding_inactive_rate,
                                                       delay, dt)

test_encoder = SpikePopulationGroupBatchToTimeEncoder(presentation_duration,
                                                      input_encoding_rate, input_encoding_inactive_rate,
                                                      delay, dt)

stdp_module = custom_stdp.BayesianSTDPClassic(output_neurons, c=1, base_mu=1, base_mu_bias=0.5,
                                              time_batch_size=stdp_time_batch_size,
                                              collect_history=True)
# stdp_module = custom_stdp.BayesianSTDPAdaptive(input_neurons, output_neurons, c=1, collect_history=True)  #todo: get this to work

# inhibition_process = OUInhibitionProcess(inhibition_increase=200, inhibition_rest=0, inhibition_tau=0.005,
#                                          noise_rest=0, noise_tau=0.005, noise_sigma=50, dt=dt)
# output_cell = StochasticOutputNeuronCell(inhibition_process=inhibition_process, dt=dt, collect_rates=True)
#
# model = BayesianSTDPModel(input_neurons, output_neurons, BinaryTimedPSP(sigma, dt),
#                           output_neuron_cell=output_cell,
#                           stdp_module=stdp_module, acc_states=False)

output_cell = EfficientStochasticOutputNeuronCell(inhibition_args=InhibitionArgs(1000, 0, 0.005),
                                                  noise_args=NoiseArgs(0, 0.005, 50),
                                                  dt=dt, collect_rates=False)

model = EfficientBayesianSTDPModel(input_neurons, output_neurons, BinaryTimedPSP(sigma, dt),
                                   multi_step_output_neuron_cell=output_cell,
                                   stdp_module=stdp_module, track_states=False)

# Model initialization
# todo: experiment with different initializations
weight_init = 2
bias_init = 2
model.linear.weight.data.fill_(weight_init)
model.linear.bias.data.fill_(bias_init)

# Training config
num_epochs = 1  # run for 1 epoch - each data sample is seen only once

loss_hist = []  # record loss over iterations
acc_hist = []  # record accuracy over iterations

state = None
total_train_output_spikes = []
total_time_ranges = [[] for _ in range(distinct_targets.shape[0])]
cumulative_counts_hist = []  # todo: might not need this
cross_entropy_hist = []
cross_entropy_paper_hist = []
time_step_hist = []

# training loop
offset = 0
for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(iter(train_loader)):
        with Timer('training loop'):
            data = rearrange(data, 'b c w h -> b (c w h)')
            input_spikes = train_encoder(data)

            time_ranges = train_encoder.get_time_ranges_for_patterns(targets,
                                                                     distinct_pattern_count=distinct_targets.shape[0],
                                                                     offset=offset)
            time_ranges_ungrouped = test_encoder.get_time_ranges(targets.shape[0], offset=offset)

            output_spikes, state = model(input_spikes, state=state, train=True)

            with Timer('metric_processing'):
                output_spikes_np = output_spikes.cpu().numpy()
                time_offset = train_encoder.get_time_offset(offset)

                total_train_output_spikes.append(output_spikes_np)
                for idx, time_range in enumerate(time_ranges):
                    total_time_ranges[idx].extend(time_range)

                with Timer('cumulative_counts'):
                    cumulative_counts = get_cumulative_counts_over_time(np.array(output_spikes_np),
                                                                        time_ranges,
                                                                        base_counts=None if i == 0 else
                                                                        cumulative_counts[-1],
                                                                        time_offset=time_offset)
                    # cumulative_counts_hist.extend(cumulative_counts)

                offset += data.shape[0]

                with Timer('metric_printing'):
                    if i % 10 == 0 or i == len(train_loader) - 1:
                        with Timer('cross_entropy'):
                            joint_probs = get_joint_probabilities_from_counts(cumulative_counts[-1])

                            cond_cross_entropy = normalized_conditional_cross_entropy(joint_probs)
                            cross_entropy_hist.append(cond_cross_entropy)

                            cond_cross_entropy_paper = normalized_conditional_cross_entropy_paper(joint_probs)
                            cross_entropy_paper_hist.append(cond_cross_entropy_paper)

                            time_step_hist.append(i * input_spikes.shape[0] * dt)

                        print(f"Epoch {epoch}, Iteration {i} \n"
                              f"Train Loss: {cond_cross_entropy:.4f}; Paper Loss: {cond_cross_entropy_paper:.4f}")

Timer.print()

torch.save(model.state_dict(), "trained_mnist_bayesian_stdp_model.pth")

# model.load_state_dict(torch.load("trained_bayesian_stdp_model.pth"))

total_train_output_spikes = np.concatenate(total_train_output_spikes, axis=0)

neuron_mapping = get_neuron_pattern_mapping(total_train_output_spikes, total_time_ranges)


output_cell.rate_tracker.is_active = True
output_cell.rate_tracker.reset()
model.state_metric.is_active = True
model.state_metric.reset()

total_time_ranges = [[] for _ in range(distinct_targets.shape[0])]
total_input_spikes = []
total_output_spikes = []

# test loop
total_acc = 0
offset = 0
for i, (data, targets) in enumerate(iter(test_loader)):
    data = rearrange(data, 'b c w h -> b (c w h)')
    input_spikes = test_encoder(data)
    total_input_spikes.append(input_spikes.cpu().numpy())

    targets_np = targets.cpu().numpy()
    time_ranges = test_encoder.get_time_ranges_for_patterns(targets_np,
                                                            distinct_pattern_count=distinct_targets.shape[0],
                                                            offset=offset)
    time_ranges_ungrouped = test_encoder.get_time_ranges(targets_np.shape[0])

    for idx, time_range in enumerate(time_ranges):
        total_time_ranges[idx].extend(time_range)

    # todo: we reset the state for every batch here, should we?
    output_spikes, _ = model(input_spikes, state=None, train=False)
    total_output_spikes.append(output_spikes.cpu().numpy())

    preds = get_predictions(output_spikes.cpu().numpy(), time_ranges_ungrouped, neuron_mapping)
    total_acc += np.mean(preds == targets_np)

    offset += data.shape[0]

print(f"Test Accuracy: {total_acc / len(test_loader) * 100:.4f}%")

# TODO: detailed evaluation; plot weights; plot bias; plot firing rates; plot stdp; plot log probabilities; plot conditional crossentropy;
# TODO: plot accuracy over training based on final pattern mapping

total_input_spikes = np.concatenate(total_input_spikes, axis=0)
total_output_spikes = np.concatenate(total_output_spikes, axis=0)

plt.plot(cross_entropy_hist, label='Crossentropy')
plt.plot(cross_entropy_paper_hist, label='Paper Crossentropy')
plt.title('Training')
plt.xlabel('Time')
plt.ylabel('Normalized Conditional Crossentropy')
plt.ylim([0, 1])

# TODO: generalize this
hist_records_between_steps = 40
plt.xticks(np.arange(0, len(cross_entropy_hist), hist_records_between_steps),
           [f"{x:.0f}s" for x in time_step_hist[::hist_records_between_steps]])
plt.legend()
plt.show()

model.state_metric.plot()

cmap = plt.get_cmap("tab10")
group_colors = [cmap(i) for i in range(distinct_targets.shape[0])]
allowed_colors = [[idx, ] for idx in neuron_mapping]

# Plot inputs
plt.figure(figsize=(10, 7))

plt.subplot(3, 1, 1)
first_ax = plt.gca()
spikes = [total_input_spikes[:, i] for i in range(total_input_spikes.shape[1])]
# raster_plot(plt.gca(), spikes)
raster_plot_multi_color(plt.gca(), spikes, total_time_ranges, group_colors)
plt.title('Input Spikes')
plt.xlabel('Time Step')
plt.ylabel('Neuron')

# Plot output spikes using a raster plot
plt.subplot(3, 1, 2)
plt.gca().sharex(first_ax)
spikes = [total_output_spikes[:, i] for i in range(total_output_spikes.shape[1])]
raster_plot_multi_color(plt.gca(), spikes, total_time_ranges, group_colors, default_color='black',
                        allowed_colors_per_train=allowed_colors)
# raster_plot_multi_color(spikes, plt.gca(), get_color_picker(("red", "green", "blue"), test_data_time_ranges))
plt.title('Output Spikes')
plt.xlabel('Time Step')
plt.ylabel('Neuron')

plt.subplot(3, 1, 3)
plt.gca().sharex(first_ax)
neuron_colors = [group_colors[idx] for idx in neuron_mapping]
output_cell.rate_tracker.plot_relative_firing_rates(plt.gca(), colors=neuron_colors)

plt.tight_layout()
plt.show()

print(model.linear.weight)
print(model.linear.bias)

stdp_module.learning_rates_tracker.plot()

# todo: add tracker for weights / biases
# todo: plots of bias convergence to log probabilities (use differently sized signals)

# todo: investigate resistance to overlapping patterns and requirement for sparseness

# todo: add background oscillations
# todo: add avg data log likelihood & conditional crossentropy measures over time
