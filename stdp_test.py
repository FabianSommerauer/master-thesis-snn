import numpy as np
from torch import Tensor
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import norse.torch as norse
import norse.torch.functional.stdp as stdp
import matplotlib.pyplot as plt

batch_size = 128
data_path = '/tmp/data/mnist'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

_ = torch.manual_seed(0)


class SpikePopulationGroupEncoder(nn.Module):
    def __init__(self, seq_length, rate=100, dt=0.001):
        super().__init__()

        self.seq_length = seq_length
        self.encoder = norse.PoissonEncoder(seq_length, f_max=rate, dt=dt)

    def forward(self, input_values: Tensor) -> Tensor:
        encoded = torch.empty(self.seq_length, *input_values.shape, 2, device=input_values.device,
                              dtype=input_values.dtype)
        for ts in range(self.seq_length):
            neuron_active = torch.stack((input_values, 1 - input_values), dim=-1)
            encoded[ts] = self.encoder(neuron_active)
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
        padded = torch.cat((encoded, torch.zeros(self.delay_shift, *encoded.shape[1:], device=encoded.device, dtype=encoded.dtype)),
                    dim=0)
        combined = padded.view(-1, *padded.shape[2:])  # todo: check if this concatenates appropriately

        # concat all neurons within timestep into single dimension
        return combined.view(*combined.shape[:-2], -1)


class BayesianSTDPModule(nn.Module):
    def __init__(self, seq_length, input_neuron_cnt, output_neuron_cnt, output_neuron, stdp_pars: stdp.STDPParameters):
        super().__init__()
        self.linear = nn.Linear(input_neuron_cnt, output_neuron_cnt, bias=True).requires_grad_(False)
        self.seq_length = seq_length
        self.output_neuron = output_neuron

        # TODO: adjust these based on custom STDP rule
        self.stdp_pars = stdp_pars
        self.t_pre = torch.tensor(1.0)
        self.t_post = torch.tensor(1.0)

    def forward(self, input_spikes: Tensor, z_state=None, stdp_state: stdp.STDPState = None, train: bool = True) \
            -> (Tensor, object, stdp.STDPState):
        with torch.no_grad():
            input_psps = self.encode(input_spikes)  # TODO

            if stdp_state is None:
                stdp_state = stdp.STDPState(self.t_pre, self.t_post)

            for ts in range(self.seq_length):
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
