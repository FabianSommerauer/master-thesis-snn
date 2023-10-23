import norse.torch as norse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from my_metrics import SpikeRateTracker, InhibitionStateTracker


# todo: check whether torch.no_grad() or detach() is needed in classes below

class SpikePopulationGroupEncoder(nn.Module):
    def __init__(self, seq_length, active_rate=100.0, inactive_rate=0.0, dt=0.001):
        super().__init__()

        self.seq_length = seq_length
        self.encoder = norse.PoissonEncoder(seq_length, f_max=active_rate, dt=dt)
        self.inactive_factor = inactive_rate / active_rate

    def forward(self, input_values: Tensor) -> Tensor:
        # assumes input values within [0,1] (ideally binary)
        neuron_active = torch.stack((input_values, (1 - input_values)), dim=-1)

        relative_rates = (1 - self.inactive_factor) * neuron_active + self.inactive_factor

        encoded = self.encoder(relative_rates)
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

    def get_time_ranges(self, pattern_count, epsilon=1e-5):
        total_len = self.base_encoder.seq_length + self.delay_shift
        # todo: check with epsilon
        time_ranges = [(i * total_len - epsilon, i * total_len + total_len - epsilon)
                       for i in range(pattern_count)]

        return time_ranges

    def get_time_ranges_for_patterns(self, pattern_order, epsilon=1e-5):
        time_ranges = self.get_time_ranges(len(pattern_order), epsilon)

        distinct_pattern_count = len(np.unique(pattern_order))
        grouped_time_ranges = [[] for _ in range(distinct_pattern_count)]
        for index, time_range in zip(pattern_order, time_ranges):
            grouped_time_ranges[index].append(time_range)

        return grouped_time_ranges


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


# todo: if this is just an object it wont be serialized properly (maybe module?)
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

        self.rate_tracker = SpikeRateTracker(is_active=collect_rates)

    def forward(self, inputs, inhibition_state=None):
        if inhibition_state is None:
            no_spike_occurrences = torch.zeros(*inputs.shape[:-1], 1, dtype=inputs.dtype, device=inputs.device)
            inhibition_state = self.inhibition_process.step(no_spike_occurrences)

        inhibition, noise = inhibition_state

        log_rates = inputs - inhibition + noise
        input_rates = torch.exp(inputs)

        # collect rates for plotting
        self.rate_tracker.update(input_rates, log_rates)

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
        self.state_metric = InhibitionStateTracker(is_active=acc_states)

        self.stdp_module = stdp_module

    def forward(self, input_spikes: Tensor, state=None, train: bool = True) \
            -> (Tensor, Tensor):
        with torch.no_grad():
            z_state = state

            seq_length = input_spikes.shape[0]
            input_psps = self.input_psp(input_spikes)

            z_out_acc = []

            for ts in range(seq_length):
                input_psp = input_psps[ts]
                z_in = self.linear(input_psp)
                z_out, z_state = self.output_neuron_cell(z_in, z_state)

                z_out_acc.append(z_out)

                # collect inhibition/noise states for plotting
                self.state_metric(z_state)

                if train:
                    new_weights, new_biases = self.stdp_module(input_psp, z_out,
                                                               self.linear.weight.data,
                                                               self.linear.bias.data)
                    self.linear.weight.data = new_weights
                    self.linear.bias.data = new_biases

            return torch.stack(z_out_acc), z_state
