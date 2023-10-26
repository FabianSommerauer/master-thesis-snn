import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import random
import torchmetrics

import custom_stdp
from my_spike_modules import *
from my_utils import spike_in_range, get_neuron_pattern_mapping, get_predictions
from my_plot_utils import raster_plot, raster_plot_multi_color, raster_plot_multi_color_per_train

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Set seed
seed = 44
random.seed(seed)
_ = torch.manual_seed(seed)
np.random.seed(seed)

# Data config
batch_size = 128
data_path = '/tmp/data/mnist'

# Data transforms
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,)),
    # ToBinaryTransform(0.5)
])

# Load data
mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

# TODO: filter data to subset of classes or subset of samples per class


# Model config
_, width, height = mnist_train.data.shape
binary_input_variable_cnt = width * height
input_neurons = binary_input_variable_cnt * 2
output_neurons = 3

dt = 0.001
sigma = 0.01

presentation_duration = 0.04
delay = 0.01

# todo: experiment with different rates (maybe different rates for train and test as well)
input_encoding_rate = 100
input_encoding_inactive_rate = 10


# Model setup
train_encoder = SpikePopulationGroupBatchToTimeEncoder(presentation_duration,
                                                       input_encoding_rate, input_encoding_inactive_rate,
                                                       delay, dt)

test_encoder = SpikePopulationGroupBatchToTimeEncoder(presentation_duration,
                                                      input_encoding_rate, input_encoding_inactive_rate,
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

# Model initialization
# todo: experiment with different initializations
weight_init = 1
bias_init = 2
model.linear.weight.data.fill_(weight_init)
model.linear.bias.data.fill_(bias_init)


# Training config
num_epochs = 1  # run for 1 epoch - each data sample is seen only once

loss_hist = []  # record loss over iterations
acc_hist = []  # record accuracy over iterations

state = None
total_train_output_spikes = None
total_time_ranges = None

# training loop
for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(iter(train_loader)):
        input_spikes = train_encoder(data)

        time_ranges = train_encoder.get_time_ranges_for_patterns(targets)
        time_ranges_ungrouped = test_encoder.get_time_ranges(len(targets))

        output_spikes, state = model(input_spikes, state=state, train=True)

        # todo: collect and combine relevant metrics here (afterwards metric trackers can be reset)

        loss_val = loss_fn(output, targets)
        loss_hist.append(loss_val.item())

        if i % 25 == 0:
            print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

            # check accuracy on a single batch
            acc = np.mean(np.argmax(output.detach().cpu().numpy(), axis=1) == targets.cpu().numpy())
            acc_hist.append(acc)
            print(f"Accuracy: {acc * 100:.2f}%\n")

neuron_mapping = get_neuron_pattern_mapping(total_train_output_spikes.cpu().numpy(), total_time_ranges)

total_acc = 0
for i, (data, targets) in enumerate(iter(test_loader)):
    input_spikes = test_encoder(data)
    # todo: we reset the state for every batch here, should we?
    output_spikes, _ = model(input_spikes, state=None, train=False)

    total_acc += np.mean(np.argmax(output.detach().cpu().numpy(), axis=1) == targets.cpu().numpy())

print(f"Test Accuracy: {total_acc / len(test_loader) * 100:.2f}%")


# TODO: detailed evaluation; plot weights; plot bias; plot firing rates; plot stdp; plot log probabilities; plot conditional crossentropy;
# TODO: plot accuracy over training based on final pattern mapping


torch.save(model.state_dict(), "trained_mnist_bayesian_stdp_model.pth")

# model.load_state_dict(torch.load("trained_bayesian_stdp_model.pth"))


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

group_colors = ("tab:orange", "tab:green", "tab:blue")
allowed_colors = [[idx, ] for idx in
                  neuron_mapping]

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

preds = get_predictions(output_spikes, test_time_ranges_ungrouped, neuron_mapping)
# todo: test if below crashes on missing predictions
acc = torchmetrics.functional.accuracy(torch.tensor(preds), test_pattern_order, task="multiclass", num_classes=3)
print(acc)

stdp_module.learning_rates_tracker.plot()

# todo: add tracker for weights / biases
# todo: plots of bias convergence to log probabilities (use differently sized signals)

# todo: investigate resistance to overlapping patterns and requirement for sparseness

# todo: add background oscillations
# todo: add avg data log likelihood & conditional crossentropy measures over time
