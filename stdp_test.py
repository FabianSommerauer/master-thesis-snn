import numpy as np
from torch import Tensor
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import norse.torch as norse
import norse.torch.functional.stdp as stdp
import matplotlib.pyplot as plt
from einops import rearrange

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
        neuron_active = torch.stack((input_values, 1 - input_values), dim=-1)
        encoded = self.encoder(neuron_active)
        return encoded


class SpikePopulationGroupBatchToTimeEncoder(nn.Module):
    def __init__(self, seq_length, rate=100, dt=0.001, delay=0.1):
        super().__init__()
        self.base_encoder = SpikePopulationGroupEncoder(seq_length, rate, dt)
        self.delay_shift = int(delay / dt)

    def forward(self, input_values: Tensor) -> Tensor:
        # shape (time, batch, input_values, neurons_per_input = 2)
        encoded = self.base_encoder(input_values)

        # move batch dimension into time dimension (concat with delay); during delay no spikes are emitted
        # also concat all neurons within timestep into single dimension
        padded = torch.cat((encoded, torch.zeros(self.delay_shift, *encoded.shape[1:], device=encoded.device, dtype=encoded.dtype)),
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
    plt.bar(x + (i+0.1) * width, input_test[:, i], width, label=f'Input {i + 1}')

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
                                                 padding=self.duration-1)

            psp = torch.clip(psp_sum, 0, 1)[..., :-(self.duration-1)]

            return rearrange(psp, '... 1 t -> t ...')


test_input_spikes = torch.tensor(np.random.randint(0, 2, (1000, 1)) * np.random.randint(0, 2, (1000, 1)) * np.random.randint(0, 2, (1000, 1)) * np.random.randint(0, 2, (1000, 1)) * np.random.randint(0, 2, (1000, 1)) * np.random.randint(0, 2, (1000, 1)))
plt.plot(test_input_spikes)
plt.plot(BinaryTimedPSP(0.01)(test_input_spikes).cpu().numpy())
plt.show()

class BayesianSTDPModule(nn.Module):
    def __init__(self, input_neuron_cnt, output_neuron_cnt,
                 input_encoder, output_neuron, stdp_pars: stdp.STDPParameters):
        super().__init__()
        self.linear = nn.Linear(input_neuron_cnt, output_neuron_cnt, bias=True).requires_grad_(False)
        self.output_neuron = output_neuron

        # TODO: adjust these based on custom STDP rule
        self.stdp_pars = stdp_pars
        self.t_pre = torch.tensor(1.0)
        self.t_post = torch.tensor(1.0)

        self.input_encoder = input_encoder

    def forward(self, input_spikes: Tensor, z_state=None, stdp_state: stdp.STDPState = None, train: bool = True) \
            -> (Tensor, object, stdp.STDPState):
        with torch.no_grad():
            seq_length = input_spikes.shape[0]
            input_psps = self.input_encoder(input_spikes)

            if stdp_state is None:
                stdp_state = stdp.STDPState(self.t_pre, self.t_post)

            for ts in range(seq_length):
                z_in = self.linear(input_psps)
                z_out, z_state = self.output_neuron(z_in, z_state)

                if train:
                    new_weights, stdp_state = stdp.stdp_step_linear(input_spikes, z_out,
                                                                    self.linear.get_parameter("weight"),
                                                                    stdp_state, self.stdp_pars,
                                                                    self.dt)

            return z_out, z_state, stdp_state


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
