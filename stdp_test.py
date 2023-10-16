import matplotlib.pyplot as plt
import norse.torch as norse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torchvision import transforms

import custom_stdp

batch_size = 128
data_path = '/tmp/data/mnist'

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

import random

random.seed(2)
_ = torch.manual_seed(2)
np.random.seed(2)


# todo: check whether torch.no_grad() or detach() is needed in classes below

class SpikePopulationGroupEncoder(nn.Module):
    def __init__(self, seq_length, active_rate=100.0, inactive_rate=0.0, dt=0.001):
        super().__init__()

        self.seq_length = seq_length
        self.encoder = norse.PoissonEncoder(seq_length, f_max=active_rate, dt=dt)
        self.inactive_factor = inactive_rate / active_rate

    def forward(self, input_values: Tensor) -> Tensor:
        # assumes input values within [0,1] (ideally binary)
        neuron_active = torch.stack((input_values, (1 - input_values) * self.inactive_factor), dim=-1)
        encoded = self.encoder(neuron_active)
        return encoded


class SpikePopulationGroupBatchToTimeEncoder(nn.Module):
    def __init__(self, presentation_duration=0.1, active_rate=100, inactive_rate=0.0, delay=0.01, dt=0.001):
        super().__init__()
        seq_length = int(presentation_duration / dt)
        self.base_encoder = SpikePopulationGroupEncoder(seq_length, active_rate, inactive_rate, dt)
        self.delay_shift = int(delay / dt)

    def forward(self, input_values: Tensor) -> Tensor:
        # shape (time, batch, input_values, neurons_per_input = 2)
        encoded = self.base_encoder(input_values)

        # move batch dimension into time dimension (concat with delay); during delay no spikes are emitted
        # also concat all neurons within timestep into single dimension
        padded = torch.cat(
            (encoded, torch.zeros(self.delay_shift, *encoded.shape[1:], device=encoded.device, dtype=encoded.dtype)),
            dim=0)
        combined = rearrange(padded, 't b ... i n -> (b t) ... (i n)')
        return combined


encoder_test = SpikePopulationGroupBatchToTimeEncoder(1000, 50, 0.001, 0.2)
input_test = torch.randint(0, 2, (4, 3))  # 4 batches, 3 inputs

# (time, spikes - 0 or 1)
encoded_test = encoder_test(input_test)


# encoded_test = torch.flip(encoded_test, dims=[-1])  # flip spikes for more intuitive visualization

# Define a function to create a raster plot for spikes
def raster_plot(spikes, ax, color='b'):
    for i, spike_train in enumerate(spikes):
        ax.eventplot(np.where(spike_train > 0)[0], lineoffsets=i, linelengths=0.7, colors=color)


# Plot inputs
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
x = np.arange(input_test.shape[0])
width = 0.2

for i in range(input_test.shape[1]):  # Iterate over inputs
    plt.bar(x + (i + 0.1) * width, input_test[:, i], width, label=f'Input {i + 1}')

plt.title('Input Test')
plt.xlabel('Batch')
plt.ylabel('Input Value')
plt.xticks(x + width * (input_test.shape[0] - 1) / 2, [f'Batch {i + 1}' for i in range(input_test.shape[0])])
plt.ylim(-0.1, 1.1)
plt.legend()

# Plot encoded spikes using a raster plot
plt.subplot(2, 1, 2)
spikes = [encoded_test[:, i].cpu().numpy() for i in range(encoded_test.shape[1])]
raster_plot(spikes, plt.gca())
plt.title('Encoded Spikes Test')
plt.xlabel('Time Step')
plt.ylabel('Neuron')

plt.tight_layout()
plt.show()


class BinaryTimedPSP(nn.Module):
    def __init__(self, sigma=0.1, dt=0.001):
        super().__init__()
        self.duration = int(sigma / dt)

    def forward(self, input_spikes: Tensor) -> Tensor:
        with torch.no_grad():
            convolvable_spikes = rearrange(input_spikes, 't ... -> ... 1 t')
            filter = torch.ones(1, 1, self.duration, device=input_spikes.device, dtype=input_spikes.dtype,
                                requires_grad=False)

            psp_sum = torch.nn.functional.conv1d(convolvable_spikes,
                                                 filter,
                                                 padding=self.duration - 1)

            psp = torch.clip(psp_sum, 0, 1)[..., :-(self.duration - 1)]

            return rearrange(psp, '... 1 t -> t ...')


test_input_spikes = torch.tensor(
    np.random.randint(0, 2, (1000, 1)) * np.random.randint(0, 2, (1000, 1)) * np.random.randint(0, 2, (
        1000, 1)) * np.random.randint(0, 2, (1000, 1)) * np.random.randint(0, 2, (1000, 1)) * np.random.randint(0, 2,
                                                                                                                (1000,
                                                                                                                 1)))
plt.plot(test_input_spikes)
plt.plot(BinaryTimedPSP(0.01)(test_input_spikes).cpu().numpy())
plt.show()


class OUInhibitionProcess(object):
    def __init__(self, inhibition_increase=3000, inhibition_rest=500, inhibition_tau=0.005,
                 noise_rest=1000, noise_tau=0.005, noise_sigma=50, dt=0.001):
        super(OUInhibitionProcess, self).__init__()

        self.inhibition_increase = inhibition_increase
        self.inhibition_rest = inhibition_rest
        # self.inhibition_tau = inhibition_tau
        self.noise_rest = noise_rest
        # self.noise_tau = noise_tau
        # self.noise_sigma = noise_sigma

        self.dt = dt
        # self.dt_sqrt = np.sqrt(dt)

        self.inhibition_decay_factor = np.exp(- dt / inhibition_tau)
        self.noise_decay_factor = np.exp(- dt / noise_tau)
        self.total_noise_sigma = noise_sigma  # todo: * np.sqrt((1. - self.noise_decay_factor ** 2) / 2. * noise_tau)

    def step(self, spike_occurrences, state=None):
        if state is None:
            inhibition = torch.zeros_like(spike_occurrences) * self.inhibition_rest
            noise = torch.ones_like(spike_occurrences) * self.noise_rest
        else:
            inhibition, noise = state

        inhibition = (self.inhibition_rest + (inhibition - self.inhibition_rest) * self.inhibition_decay_factor
                      + spike_occurrences * self.inhibition_increase)

        # Euler approximation of Ornstein-Uhlenbeck process
        # noise = (1 - self.decay_rate * self.dt) * noise \
        #             + self.decay_sigma * self.dt_sqrt * torch.normal(0, 1, noise.shape)

        # more direct approximation based on
        # https://github.com/cantaro86/Financial-Models-Numerical-Methods/blob/master/6.1%20Ornstein-Uhlenbeck%20process%20and%20applications.ipynb
        noise = (self.noise_rest + (noise - self.noise_rest) * self.noise_decay_factor
                 + self.total_noise_sigma * torch.normal(0, 1, noise.shape))

        return inhibition, noise


# based on https://github.com/pytorch/pytorch/issues/30968#issuecomment-859084590
def efficient_multinomial(r):
    return (r.cumsum(-1) >= torch.rand(r.shape[:-1])[..., None]).byte().argmax(-1)


class StochasticOutputNeuronCell(nn.Module):
    def __init__(self, inhibition_process: OUInhibitionProcess, dt=0.001, collect_rates=False):
        super(StochasticOutputNeuronCell, self).__init__()

        self.inhibition_process = inhibition_process
        self.dt = dt
        self.log_dt = np.log(dt)

        self.collect_rates = collect_rates
        self.input_rates_history = []
        self.log_rates_history = []

    def forward(self, inputs, inhibition_state=None):
        if inhibition_state is None:
            no_spike_occurrences = torch.zeros(*inputs.shape[:-1], 1, dtype=inputs.dtype, device=inputs.device)
            inhibition_state = self.inhibition_process.step(no_spike_occurrences)

        inhibition, noise = inhibition_state

        log_rates = inputs - inhibition + noise
        input_rates = torch.exp(inputs)

        if self.collect_rates:
            self.input_rates_history.append(input_rates)
            self.log_rates_history.append(log_rates)

        # more numerically stable to utilize log
        log_total_rate = torch.logsumexp(log_rates, -1, keepdim=True)

        rand_val = torch.rand(
            *log_total_rate.shape,
            device=log_total_rate.device,
        )

        # check rand_val < total_rate * dt  (within log range)
        spike_occurred = torch.log(rand_val) < log_total_rate + self.log_dt

        # here we only have to deal with input_rates as inhibition + noise cancels out
        # (makes process more numerically stable)
        rel_probs = input_rates / torch.sum(input_rates, dim=-1, keepdim=True)
        spike_location = efficient_multinomial(rel_probs)

        out_spike_locations = F.one_hot(spike_location, num_classes=input_rates.shape[-1])

        # ensures that only one output neuron can fire at a time
        out_spikes = (out_spike_locations * spike_occurred).to(dtype=inputs.dtype)

        # increase inhibition if a spike occured
        inhibition_state = self.inhibition_process.step(spike_occurred, inhibition_state)

        return out_spikes, inhibition_state


class BayesianSTDPModel(nn.Module):
    def __init__(self, input_neuron_cnt, output_neuron_cnt,
                 input_psp, output_neuron_cell,
                 stdp_module,
                 acc_states=False):
        super().__init__()
        self.linear = nn.Linear(input_neuron_cnt, output_neuron_cnt, bias=True).requires_grad_(False)
        self.output_neuron_cell = output_neuron_cell

        self.input_psp = input_psp
        self.acc_states = acc_states

        self.stdp_module = stdp_module

    def forward(self, input_spikes: Tensor, state=None, train: bool = True) \
            -> (Tensor, Tensor):
        with torch.no_grad():
            z_state = state

            seq_length = input_spikes.shape[0]
            input_psps = self.input_psp(input_spikes)

            z_out_acc = []
            if self.acc_states:
                state_acc = []

            for ts in range(seq_length):
                input_psp = input_psps[ts]
                z_in = self.linear(input_psp)
                z_out, z_state = self.output_neuron_cell(z_in, z_state)

                z_out_acc.append(z_out)
                if self.acc_states:
                    state_acc.append(z_state)

                if train:
                    new_weights, new_biases = self.stdp_module(input_psp, z_out,
                                                               self.linear.weight.data,
                                                               self.linear.bias.data)
                    self.linear.weight.data = new_weights
                    self.linear.bias.data = new_biases

            return torch.stack(z_out_acc), state_acc if self.acc_states else z_state


class ToBinaryTransform(object):
    def __init__(self, thresh):
        self.thresh = thresh

    def __call__(self, data: Tensor):
        return (data > self.thresh).to(data.dtype)


class MaxPotentialDecode(nn.Module):
    def __init__(self):
        super(MaxPotentialDecode, self).__init__()

    def forward(self, membrane_potential: Tensor) -> Tensor:
        return membrane_potential.max(dim=0).values


class RateDecode(nn.Module):
    def __init__(self):
        super(RateDecode, self).__init__()

    def forward(self, spk_trns: Tensor) -> Tensor:
        return spk_trns.mean(dim=0)


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


duplications = 3
binary_input_variable_cnt = 4
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
                                                 input_encoding_rate*2, 0,
                                                 delay, dt)

stdp_module = custom_stdp.BayesianSTDPClassic(output_neurons, c=1, base_mu=1, base_mu_bias=0.5, collect_history=True)
# stdp_module = custom_stdp.BayesianSTDPAdaptive(input_neurons, output_neurons, c=1, collect_history=True)  #todo: get this to work

inhibition_process = OUInhibitionProcess(inhibition_increase=3000, inhibition_rest=0, inhibition_tau=0.005,
                                         noise_rest=0, noise_tau=0.005, noise_sigma=50, dt=dt)
output_cell = StochasticOutputNeuronCell(
    inhibition_process=inhibition_process, dt=dt)

model = BayesianSTDPModel(input_neurons, output_neurons, BinaryTimedPSP(sigma, dt),
                          output_neuron_cell=output_cell,
                          stdp_module=stdp_module, acc_states=True)

# todo
data = torch.randint(0, 2, (output_neurons, binary_input_variable_cnt))  # 3 batches of 5 bit patterns
print(data)

train_data = data.repeat(666, duplications)  # data.repeat(2000, 3)
test_data = data.repeat(4, duplications)
input_spikes = train_encoder(train_data)
_, _ = model(input_spikes, train=True)

output_cell.collect_rates = True
input_spikes = test_encoder(test_data)
output_spikes, output_states = model(input_spikes, train=False)

print(input_spikes)
print(output_spikes)
print(output_states)

inhibition = [state[0] - state[1] for state in output_states]
plt.plot(inhibition)
plt.show()

# inhibition = [state[0] for state in output_states]
# noise = [state[1] for state in output_states]
# plt.plot(inhibition)
# plt.show()
#
# plt.plot(noise)
# plt.show()

# Plot inputs
plt.figure(figsize=(10, 7))

plt.subplot(3, 1, 1)
spikes = [input_spikes[:, i].cpu().numpy() for i in range(input_spikes.shape[1])]
raster_plot(spikes, plt.gca())
plt.title('Input Spikes')
plt.xlabel('Time Step')
plt.ylabel('Neuron')

# Plot output spikes using a raster plot
plt.subplot(3, 1, 2)
spikes = [output_spikes[:, i].cpu().numpy() for i in range(output_spikes.shape[1])]
raster_plot(spikes, plt.gca())
plt.title('Output Spikes')
plt.xlabel('Time Step')
plt.ylabel('Neuron')

plt.subplot(3, 1, 3)
input_rates = torch.stack(output_cell.input_rates_history, dim=0).cpu().numpy()
normalized_rates = input_rates / np.sum(input_rates, axis=-1, keepdims=True)
for i in range(output_neurons):
    plt.plot(normalized_rates[:, i], label=f'Rate Output neuron {i}')
plt.title('Relative Firing Rates')
plt.xlabel('Time Step')
plt.ylabel('Rate')
plt.legend()

plt.tight_layout()
plt.show()

print(model.linear.weight)
print(model.linear.bias)

mu_w_history = torch.stack(stdp_module.mu_w_history)
mu_b_history = torch.stack(stdp_module.mu_b_history)

# todo: this has different interpretations for different STDP modules -> fix
plt.plot(torch.mean(mu_w_history, -1)[:, 0], label='mu_w 0')
plt.plot(torch.mean(mu_w_history, -1)[:, 1], label='mu_w 1')
plt.plot(torch.mean(mu_w_history, -1)[:, 2], label='mu_w 2')
plt.plot(torch.mean(mu_b_history, -1)[:], label='mu_b')
plt.yscale('log')
plt.legend()
plt.show()


# todo: improve plots above (colour spikes based on group (separate them appropriately) + whether correctly predicted)
# todo: plots of relative rates for each neuron based on input
# todo: plots of bias convergence to log probabilities (use differently sized signals)