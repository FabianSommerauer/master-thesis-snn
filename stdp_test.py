import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import random

import custom_stdp
from my_spike_modules import *
from my_utils import spike_in_range, get_neuron_pattern_mapping
from my_plot_utils import raster_plot, raster_plot_multi_color, raster_plot_multi_color_per_train

batch_size = 128
data_path = '/tmp/data/mnist'

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

seed = 44
random.seed(seed)
_ = torch.manual_seed(seed)
np.random.seed(seed)

# Spike encoding test - TODO

# encoder_test = SpikePopulationGroupBatchToTimeEncoder(1000, 50, 0.001, 0.2)
# input_test = torch.randint(0, 2, (4, 3))  # 4 batches, 3 inputs
#
# # (time, spikes - 0 or 1)
# encoded_test = encoder_test(input_test)
#
#
# # encoded_test = torch.flip(encoded_test, dims=[-1])  # flip spikes for more intuitive visualization


# # Plot inputs
# plt.figure(figsize=(10, 5))
# plt.subplot(2, 1, 1)
# x = np.arange(input_test.shape[0])
# width = 0.2
#
# for i in range(input_test.shape[1]):  # Iterate over inputs
#     plt.bar(x + (i + 0.1) * width, input_test[:, i], width, label=f'Input {i + 1}')
#
# plt.title('Input Test')
# plt.xlabel('Batch')
# plt.ylabel('Input Value')
# plt.xticks(x + width * (input_test.shape[0] - 1) / 2, [f'Batch {i + 1}' for i in range(input_test.shape[0])])
# plt.ylim(-0.1, 1.1)
# plt.legend()
#
# # Plot encoded spikes using a raster plot
# plt.subplot(2, 1, 2)
# spikes = [encoded_test[:, i].cpu().numpy() for i in range(encoded_test.shape[1])]
# raster_plot(spikes, plt.gca())
# plt.title('Encoded Spikes Test')
# plt.xlabel('Time Step')
# plt.ylabel('Neuron')
#
# plt.tight_layout()
# plt.show()


# Test Binary Timed PSP

# test_input_spikes = torch.tensor(
#     np.random.randint(0, 2, (1000, 1)) * np.random.randint(0, 2, (1000, 1)) * np.random.randint(0, 2, (
#         1000, 1)) * np.random.randint(0, 2, (1000, 1)) * np.random.randint(0, 2, (1000, 1)) * np.random.randint(0, 2,
#                                                                                                                 (1000,
#                                                                                                                  1)))
# plt.plot(test_input_spikes)
# plt.plot(BinaryTimedPSP(0.01)(test_input_spikes).cpu().numpy())
# plt.show()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,)),
    # ToBinaryTransform(0.5)
])

# mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
# mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

# train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)


duplications = 1
train_repeats = 666
test_repeats = 4

binary_input_variable_cnt = 30
input_neurons = binary_input_variable_cnt * 2 * duplications
output_neurons = 3

dt = 0.001
sigma = 0.01
rate_multiplier = 1

presentation_duration = 0.04
delay = 0.01
input_encoding_rate = 100
input_encoding_inactive_rate = 10

train_encoder = SpikePopulationGroupBatchToTimeEncoder(presentation_duration,
                                                       input_encoding_rate, input_encoding_inactive_rate,
                                                       delay, dt)

test_encoder = SpikePopulationGroupBatchToTimeEncoder(presentation_duration,
                                                      input_encoding_rate * 2, 0,
                                                      delay, dt)

stdp_module = custom_stdp.BayesianSTDPClassic(output_neurons, c=1, base_mu=1, base_mu_bias=0.5, collect_history=True)
# stdp_module = custom_stdp.BayesianSTDPAdaptive(input_neurons, output_neurons, c=1, collect_history=True)  #todo: get this to work

inhibition_process = OUInhibitionProcess(inhibition_increase=3000, inhibition_rest=0, inhibition_tau=0.005,
                                         noise_rest=0, noise_tau=0.005, noise_sigma=50, dt=dt)
output_cell = StochasticOutputNeuronCell(
    inhibition_process=inhibition_process, dt=dt)

model = BayesianSTDPModel(input_neurons, output_neurons, BinaryTimedPSP(sigma, dt),
                          output_neuron_cell=output_cell,
                          stdp_module=stdp_module, acc_states=False)

# todo
data = torch.multinomial(torch.tensor([0.85, 0.15]), output_neurons * binary_input_variable_cnt, replacement=True)
data = data.reshape(output_neurons, binary_input_variable_cnt)
# data = torch.randint(0, 2, (output_neurons, binary_input_variable_cnt))  # 3 batches of 5 bit patterns
print(data)

pattern_duration = presentation_duration + delay

pattern_order = torch.tensor([0, 1, 2])
train_pattern_order = pattern_order.repeat(train_repeats)
test_pattern_order = pattern_order.repeat(test_repeats)


train_data = data.repeat(train_repeats, duplications)  # data.repeat(2000, 3)
test_data = data.repeat(test_repeats, duplications)
input_spikes = train_encoder(train_data)

train_time_ranges = train_encoder.get_time_ranges_for_patterns(train_pattern_order)
test_time_ranges = test_encoder.get_time_ranges_for_patterns(test_pattern_order)


# _, _ = model(input_spikes, train=True)

# todo: check what is actually saved here
# torch.save(model.state_dict(), "trained_bayesian_stdp_model.pth")

model.load_state_dict(torch.load("trained_bayesian_stdp_model.pth"))


output_cell.rate_tracker.is_active = True
output_cell.rate_tracker.reset()
model.state_metric.is_active = True
model.state_metric.reset()

input_spikes = test_encoder(test_data)
output_spikes, _ = model(input_spikes, train=False)

# todo: this should be done on the train data (maybe only last few patterns)
neuron_mapping = get_neuron_pattern_mapping(output_spikes.cpu().numpy(), test_time_ranges)

print(input_spikes)
print(output_spikes)

model.state_metric.plot()

# inhibition = [state[0] for state in output_states]
# noise = [state[1] for state in output_states]
# plt.plot(inhibition)
# plt.show()
#
# plt.plot(noise)
# plt.show()

group_colors = ("tab:orange", "tab:green", "tab:blue")
allowed_colors = [[idx,] for idx in neuron_mapping]  # todo: find these by finding the correct output neuron for each pattern

# Plot inputs
plt.figure(figsize=(10, 7))

plt.subplot(3, 1, 1)
first_ax = plt.gca()
spikes = [input_spikes[:, i].cpu().numpy() for i in range(input_spikes.shape[1])]
# raster_plot(plt.gca(), spikes)
raster_plot_multi_color(plt.gca(), spikes, test_time_ranges, group_colors)
plt.title('Input Spikes')
plt.xlabel('Time Step')
plt.ylabel('Neuron')

# Plot output spikes using a raster plot
plt.subplot(3, 1, 2)
plt.gca().sharex(first_ax)
spikes = [output_spikes[:, i].cpu().numpy() for i in range(output_spikes.shape[1])]
raster_plot_multi_color(plt.gca(), spikes, test_time_ranges, group_colors, default_color='black',
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

# todo: improve plots above (colour spikes based on group (separate them appropriately) + whether correctly predicted)
# todo: plots of relative rates for each neuron based on input
# todo: plots of bias convergence to log probabilities (use differently sized signals)

# todo: investigate resistance to overlapping patterns and requirement for sparseness
