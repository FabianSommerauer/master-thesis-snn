import numpy as np
from torch import Tensor
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import norse.torch as norse
import norse.torch.functional.stdp as stdp
import matplotlib.pyplot as plt
from einops import rearrange
import custom_stdp

batch_size = 128
data_path = '/tmp/data/mnist'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

_ = torch.manual_seed(0)


# todo: check whether torch.no_grad() or detach() is needed in classes below

class SpikePopulationGroupEncoder(nn.Module):
    def __init__(self, seq_length, rate=100, dt=0.001):
        super().__init__()

        self.seq_length = seq_length
        self.encoder = norse.PoissonEncoder(seq_length, f_max=rate, dt=dt)

    def forward(self, input_values: Tensor) -> Tensor:
        # assumes input values within [0,1] (ideally binary)
        neuron_active = torch.stack((input_values, 1 - input_values), dim=-1)
        encoded = self.encoder(neuron_active)
        return encoded


class SpikePopulationGroupBatchToTimeEncoder(nn.Module):
    def __init__(self, presentation_duration=0.1, rate=100,delay=0.01, dt=0.001):
        super().__init__()
        seq_length = int(presentation_duration / dt)
        self.base_encoder = SpikePopulationGroupEncoder(seq_length, rate, dt)
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
                                                                                                            (1000, 1)))
plt.plot(test_input_spikes)
plt.plot(BinaryTimedPSP(0.01)(test_input_spikes).cpu().numpy())
plt.show()


class StochasticOutputNeuronCell(nn.Module):
    def __init__(self, inhibition_increase=1.0, decay_rate=0.5, decay_sigma=0.3, dt=0.001):
        super(StochasticOutputNeuronCell, self).__init__()

        self.inhibition_increase = inhibition_increase
        self.decay_rate = decay_rate
        self.decay_sigma = decay_sigma
        self.dt = dt
        self.dt_sqrt = np.sqrt(dt)

    def forward(self, inputs, inhibition=None):
        if inhibition is None:
            inhibition = torch.zeros(*inputs.shape[:-1], 1, dtype=inputs.dtype, device=inputs.device)
        else:
            # Euler approximation of Ornstein-Uhlenbeck process
            inhibition = (1 - self.decay_rate * self.dt) * inhibition \
                         + self.decay_sigma * self.dt_sqrt * torch.normal(0, 1, inhibition.shape)

            # todo: maybe use more direct solution from
            # https://github.com/cantaro86/Financial-Models-Numerical-Methods/blob/master/6.1%20Ornstein-Uhlenbeck%20process%20and%20applications.ipynb

        rates = torch.exp(inputs - inhibition)

        rand_vals = torch.rand(
            *inputs.shape,
            device=inputs.device,
        )
        min_vals, min_indices = torch.min(rand_vals, -1)

        # prefer spikes from neurons with lowest random value
        out_spike_locations = F.one_hot(min_indices, num_classes=inputs.shape[-1])
        spike_occurred = rand_vals < self.dt * rates

        # ensures that only one output neuron can fire at a time
        out_spikes = (out_spike_locations * spike_occurred).to(dtype=inputs.dtype)

        # increase inhibition if a spike occured
        inhibition += torch.max(out_spikes, -1, keepdim=True).values * self.inhibition_increase

        return out_spikes, inhibition


class BayesianSTDPModel(nn.Module):
    def __init__(self, input_neuron_cnt, output_neuron_cnt,
                 input_psp, output_neuron_cell,
                 stdp_c=1, stdp_mu=0.1,
                 acc_states=False):
        super().__init__()
        self.linear = nn.Linear(input_neuron_cnt, output_neuron_cnt, bias=True).requires_grad_(False)
        self.output_neuron_cell = output_neuron_cell

        self.input_psp = input_psp
        self.acc_states = acc_states

        self.stdp_c = stdp_c
        self.stdp_mu = stdp_mu

    def forward(self, input_spikes: Tensor, z_state=None, train: bool = True) \
            -> (Tensor, Tensor):
        with torch.no_grad():
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
                    new_weights, new_biases = custom_stdp.apply_bayesian_stdp(input_psp, z_out,
                                                                              self.linear.weight.data,
                                                                              self.linear.bias.data,
                                                                              self.stdp_c,
                                                                              self.stdp_mu)
                    self.linear.weight.data = new_weights
                    self.linear.bias.data = new_biases

            return torch.stack(z_out_acc), torch.stack(state_acc) if self.acc_states else z_state


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

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)


binary_input_variable_cnt = 5
input_neurons = binary_input_variable_cnt * 2
output_neurons = 3

dt = 0.001
sigma = 0.01

presentation_duration = 0.1
delay = 0.02
input_encoding_rate = 500

encoder = SpikePopulationGroupBatchToTimeEncoder(presentation_duration, input_encoding_rate, delay, dt)

model = BayesianSTDPModel(input_neurons, output_neurons, BinaryTimedPSP(sigma, dt), StochasticOutputNeuronCell(dt=dt),
                          stdp_c=1, stdp_mu=0.1, acc_states=True)

# todo
data = torch.randint(0, 2, (output_neurons, binary_input_variable_cnt))   # 3 batches of 5 bit patterns
input_spikes = encoder(data)
output_spikes, output_states = model(input_spikes, train=True)

# todo: plot everything