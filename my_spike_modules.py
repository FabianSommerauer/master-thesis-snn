from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from deprecation import deprecated
from einops import rearrange
from torch import Tensor

from my_timing_utils import measure_time, Timer
from my_trackers import SpikeRateTracker, InhibitionStateTracker, WeightsTracker


@dataclass
class InhibitionArgs:
    inhibition_increase: float
    inhibition_rest: float
    inhibition_tau: float


@dataclass
class NoiseArgs:
    noise_rest: float
    noise_tau: float
    noise_sigma: float


@dataclass
class BackgroundOscillationArgs:
    osc_amplitude: float
    osc_freq: float
    osc_phase: float


class LogFiringRateCalculationMode(Enum):
    Default = 0  # inputs + noise - inhibition
    IgnoreInputs = 1  # noise - inhibition
    ExpectedInputCorrected = 2  # inputs - mean(inputs) + noise - inhibition


# todo: check whether torch.no_grad() or detach() is needed in classes below

class SpikePopulationGroupEncoder(nn.Module):
    def __init__(self, seq_length, active_rate=100.0, inactive_rate=0.0, dt=0.001):
        super().__init__()

        self.seq_length = seq_length
        self.active_rate = active_rate
        self.inactive_rate = inactive_rate
        self.dt = dt

    def forward(self, input_values: Tensor, rate_modulation: Tensor = None) -> Tensor:
        # assumes input values within [0,1] (ideally binary)
        neuron_active = torch.stack((input_values, (1 - input_values)), dim=-1)

        rates = (1 - neuron_active) * self.inactive_rate + neuron_active * self.active_rate

        if rate_modulation is not None:
            rates = rates[None, ...] * rate_modulation[..., None, None]

        encoded = (torch.rand(
            self.seq_length,
            *neuron_active.shape,
            device=rates.device
        ).float() < self.dt * rates).float()
        return encoded


class SpikePopulationGroupBatchToTimeEncoder(nn.Module):
    def __init__(self, presentation_duration=0.1, active_rate=100, inactive_rate=0.0, delay=0.01, dt=0.001,
                 background_oscillation_args: BackgroundOscillationArgs = None):
        super().__init__()
        self.dt = dt
        self.seq_length = int(presentation_duration / dt)
        self.base_encoder = SpikePopulationGroupEncoder(self.seq_length, active_rate, inactive_rate, dt)
        self.delay_shift = int(delay / dt)

        self.background_oscillation_active = background_oscillation_args is not None
        if self.background_oscillation_active:
            self.background_oscillation_amplitude = background_oscillation_args.osc_amplitude
            self.background_oscillation_freq = background_oscillation_args.osc_freq
            self.background_oscillation_phase = background_oscillation_args.osc_phase

    @measure_time
    def forward(self, input_values: Tensor, start_phase: Tensor = None) -> tuple[Tensor, Tensor] | Tensor:
        if self.background_oscillation_active:
            phase = self.background_oscillation_freq * torch.arange(self.seq_length) * self.dt

            batch_offsets = (torch.arange(input_values.shape[0], device=input_values.device)
                             * self.get_shift_between_patterns() * self.dt
                             * self.background_oscillation_freq)

            phase = phase[..., None] + batch_offsets[None, ...]

            next_start_phase = (self.get_shift_between_patterns() * input_values.shape[0] * self.dt
                                * self.background_oscillation_freq)

            phase *= 2 * torch.pi
            next_start_phase *= 2 * torch.pi

            if start_phase is not None:
                phase += start_phase
                next_start_phase += start_phase
            else:
                phase += self.background_oscillation_phase
                next_start_phase += self.background_oscillation_phase

            rate_modulation = self.background_oscillation_amplitude * (1 + torch.zeros_like(phase))
        else:
            rate_modulation = None
            next_start_phase = None

        # shape (time, batch, input_values, neurons_per_input = 2)
        encoded = self.base_encoder(input_values, rate_modulation)

        # move batch dimension into time dimension (concat with delay); during delay no spikes are emitted
        # also concat all neurons within timestep into single dimension
        padded = torch.cat(
            (encoded, torch.zeros(self.delay_shift, *encoded.shape[1:], device=encoded.device, dtype=encoded.dtype)),
            dim=0)
        combined = rearrange(padded, 't b ... i n -> (b t) ... (i n)')

        return combined, next_start_phase

    @measure_time
    def get_time_ranges(self, pattern_count, epsilon=1e-5, offset=0):
        total_len = self.seq_length + self.delay_shift

        time_ranges = [((i + offset) * total_len - epsilon,
                        (i + offset) * total_len + self.seq_length + epsilon)
                       for i in range(pattern_count)]

        return time_ranges

    @measure_time
    def get_time_ranges_for_patterns(self, pattern_order, distinct_pattern_count=None, epsilon=1e-5, offset=0):
        time_ranges = self.get_time_ranges(len(pattern_order), epsilon, offset)

        if distinct_pattern_count is None:
            distinct_pattern_count = len(np.unique(pattern_order))
        grouped_time_ranges = [[] for _ in range(distinct_pattern_count)]
        for index, time_range in zip(pattern_order, time_ranges):
            grouped_time_ranges[index].append(time_range)

        return grouped_time_ranges

    def get_time_for_offset(self, offset):
        return offset * (self.seq_length + self.delay_shift)

    def get_shift_between_patterns(self):
        return self.seq_length + self.delay_shift


class BinaryTimedPSP(nn.Module):
    def __init__(self, sigma=0.1, dt=0.001):
        super().__init__()
        self.duration = int(sigma / dt)

    @measure_time
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


def filter_response_over_time(data: Tensor, filter: Tensor):
    """Convolves over time dimension (first dimension) with given filter (in reverse order).
    Ensures that output is properly aligned in time.

    Args:
        data: shape (time, ...)
        filter: shape (filter_size)
    """
    shape = data.shape

    reversed_filter = torch.flip(filter, dims=(-1,))

    compacted_data = torch.reshape(data, (data.shape[0], -1))
    convolvable_data = rearrange(compacted_data, 't r -> r 1 t')
    convolved = torch.nn.functional.conv1d(convolvable_data,
                                           reversed_filter[None, None, :],
                                           padding=filter.shape[-1] - 1)
    relevant_convolved = convolved[..., :-(filter.shape[-1] - 1)]
    result = rearrange(relevant_convolved, 'r 1 t -> t r')
    return torch.reshape(result, shape)


@deprecated(details="This functionality has moved into EfficientStochasticOutputNeuronCell")
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

    @measure_time
    def step(self, spike_occurrences, state=None):
        if state is None:
            inhibition = torch.zeros_like(spike_occurrences) * self.inhibition_rest
            noise = torch.ones_like(spike_occurrences) * self.noise_rest
        else:
            inhibition, noise = state

        with Timer('inhibition'):
            inhibition = (self.inhibition_rest + (inhibition - self.inhibition_rest) * self.inhibition_decay_factor
                          + spike_occurrences * self.inhibition_increase)

        # Euler approximation of Ornstein-Uhlenbeck process
        # noise = (1 - self.decay_rate * self.dt) * noise \
        #             + self.decay_sigma * self.dt_sqrt * torch.normal(0, 1, noise.shape)

        with Timer('noise'):
            # more direct approximation based on
            # https://github.com/cantaro86/Financial-Models-Numerical-Methods/blob/master/6.1%20Ornstein-Uhlenbeck%20process%20and%20applications.ipynb
            noise = (self.noise_rest + (noise - self.noise_rest) * self.noise_decay_factor
                     + self.total_noise_sigma * torch.normal(0, 1, noise.shape))

        return inhibition, noise


# based on https://github.com/pytorch/pytorch/issues/30968#issuecomment-859084590
def efficient_multinomial(r):
    return (r.cumsum(-1) >= torch.rand(r.shape[:-1])[..., None]).byte().argmax(-1)


@deprecated(details="Use EfficientStochasticOutputNeuronCell instead.")
class StochasticOutputNeuronCell(nn.Module):
    def __init__(self, inhibition_process: OUInhibitionProcess, dt=0.001, collect_rates=False):
        super(StochasticOutputNeuronCell, self).__init__()

        self.inhibition_process = inhibition_process
        self.dt = dt
        self.log_dt = np.log(dt)

        self.rate_tracker = SpikeRateTracker(is_active=collect_rates)

    @measure_time
    def forward(self, inputs, inhibition_state=None):
        if inhibition_state is None:
            no_spike_occurrences = torch.zeros(*inputs.shape[:-1], 1, dtype=inputs.dtype, device=inputs.device)
            inhibition_state = self.inhibition_process.step(no_spike_occurrences)

        with Timer('rate_calc'):
            inhibition, noise = inhibition_state

            log_rates = inputs - inhibition + noise
            relative_input_rates = torch.exp((inputs - torch.logsumexp(inputs, dim=-1, keepdim=True))/50.)

            with Timer('rate_track'):
                # collect rates for plotting
                self.rate_tracker.update(relative_input_rates, log_rates)

            # more numerically stable to utilize log
            log_total_rate = torch.logsumexp(log_rates, -1, keepdim=True)

        with Timer('spike_gen'):
            rand_val = torch.rand(
                *log_total_rate.shape,
                device=log_total_rate.device,
            )

            # check rand_val < total_rate * dt  (within log range)
            spike_occurred = torch.log(rand_val) < log_total_rate + self.log_dt

            # here we only have to deal with input_rates as inhibition + noise cancels out
            # (makes process more numerically stable)
            # rel_probs = input_rates / torch.sum(input_rates, dim=-1, keepdim=True)  (should already be normalized due to logsumexp)
            spike_location = efficient_multinomial(relative_input_rates)

            out_spike_locations = F.one_hot(spike_location, num_classes=relative_input_rates.shape[-1])

            # ensures that only one output neuron can fire at a time
            out_spikes = (out_spike_locations * spike_occurred).to(dtype=inputs.dtype)

        # increase inhibition if a spike occured
        inhibition_state = self.inhibition_process.step(spike_occurred, inhibition_state)

        return out_spikes, inhibition_state


@deprecated(details="Use EfficientBayesianSTDPModel instead")
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

    @measure_time
    def forward(self, input_spikes: Tensor, state=None, train: bool = True, batched_update: bool = False) \
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

                with Timer('state_track'):
                    # collect inhibition/noise states for plotting
                    self.state_metric.update(z_state)

                if train and not batched_update:
                    with Timer('stdp'):
                        new_weights, new_biases = self.stdp_module(input_psp, z_out,
                                                                   self.linear.weight.data,
                                                                   self.linear.bias.data)
                        self.linear.weight.data = new_weights
                        self.linear.bias.data = new_biases

            z_out_stack = torch.stack(z_out_acc)
            if train and batched_update:
                with Timer('stdp'):
                    new_weights, new_biases = self.stdp_module(input_psps, z_out_stack,
                                                               self.linear.weight.data,
                                                               self.linear.bias.data)
                    self.linear.weight.data = new_weights
                    self.linear.bias.data = new_biases

            return z_out_stack, z_state


# Efficient implementations of the above classes

class EfficientStochasticOutputNeuronCell(nn.Module):
    def __init__(self, inhibition_args: InhibitionArgs, noise_args: NoiseArgs,
                 background_oscillation_args: BackgroundOscillationArgs = None,
                 dt=0.001, log_firing_rate_calc_mode=LogFiringRateCalculationMode.Default,
                 collect_rates=False):
        super(EfficientStochasticOutputNeuronCell, self).__init__()

        self.inhibition_args = inhibition_args
        self.noise_args = noise_args
        self.dt = dt
        self.log_dt = np.log(dt)

        self.rate_tracker = SpikeRateTracker(is_active=collect_rates, is_batched=True)

        max_noise_filter_duration = 10 * self.noise_args.noise_tau
        self.noise_filter = torch.exp(-torch.arange(0, max_noise_filter_duration, self.dt) / self.noise_args.noise_tau)

        self.inhibition_rest = self.inhibition_args.inhibition_rest
        self.inhibition_increase = self.inhibition_args.inhibition_increase
        self.inhibition_factor = np.exp(-self.dt / self.inhibition_args.inhibition_tau)

        self.log_firing_rate_calc_mode = log_firing_rate_calc_mode

        if background_oscillation_args is not None:
            self.background_oscillation_active = True
            self.background_oscillation_amplitude = background_oscillation_args.osc_amplitude
            self.background_oscillation_freq = background_oscillation_args.osc_freq
            self.background_oscillation_phase = background_oscillation_args.osc_phase
        else:
            self.background_oscillation_active = False

    @measure_time
    def forward(self, inputs, state=None):
        with Timer('inhibition'):
            noise_base = torch.ones(*inputs.shape[:-1], 1,
                                    dtype=inputs.dtype, device=inputs.device) * self.noise_args.noise_rest
            noise_rand = torch.normal(0.0, self.noise_args.noise_sigma, noise_base.shape,
                                      device=noise_base.device, dtype=noise_base.dtype)

            if state is None:
                last_inhibition = torch.zeros(1, dtype=inputs.dtype, device=inputs.device)

                if self.background_oscillation_active:
                    last_phase = self.background_oscillation_phase
            if state is not None:
                if self.background_oscillation_active:
                    last_inhibition, last_noise, last_phase = state
                else:
                    last_inhibition, last_noise = state
                noise_rand[0] += last_noise

            noise = filter_response_over_time(noise_rand, self.noise_filter)

        with Timer('background_oscillation'):
            if self.background_oscillation_active:
                phase = (self.background_oscillation_freq * torch.arange(1, inputs.shape[0] + 1)
                         * self.dt * 2 * torch.pi) + last_phase
                osc = self.background_oscillation_amplitude * torch.sin(phase)

        with Timer('rate_calc'):
            match self.log_firing_rate_calc_mode:
                case LogFiringRateCalculationMode.Default:
                    log_rates_wo_inhibition = inputs + noise
                case LogFiringRateCalculationMode.IgnoreInputs:
                    # equivalent to using inputs - torch.logsumexp(inputs, dim=-1, keepdim=True) + noise
                    log_rates_wo_inhibition = noise
                case LogFiringRateCalculationMode.ExpectedInputCorrected:
                    log_rates_wo_inhibition = inputs - torch.mean(inputs, dim=-1, keepdim=True) + noise

            if self.background_oscillation_active:
                log_rates_wo_inhibition += osc[:, None]

            relative_input_rates = torch.exp(inputs - torch.logsumexp(inputs, dim=-1, keepdim=True))

            # more numerically stable to utilize log
            # (inhibition is constant so we can pull it out of the logsumexp)
            log_total_rate_wo_inhibition = torch.logsumexp(log_rates_wo_inhibition, -1, keepdim=True)

        with Timer('spike_loc_gen'):
            # here we only have to deal with input_rates as inhibition + noise cancels out
            # (makes process more numerically stable)
            spike_location = efficient_multinomial(relative_input_rates)

            out_spike_locations = F.one_hot(spike_location, num_classes=relative_input_rates.shape[-1])

        with Timer('spike_gen'):
            rand_val = torch.rand(
                *log_total_rate_wo_inhibition.shape,
                device=log_total_rate_wo_inhibition.device,
            )

            # would check rand_val < total_rate * dt  (within log range)
            # however inhibition is not yet determined so we have to use the upper threshold
            inhibition_upper_threshold = log_total_rate_wo_inhibition + self.log_dt - torch.log(rand_val)

            with Timer('inhibition_calc'):
                inhibition, spike_occurred = self.get_spike_occurrences_and_inhibition(inhibition_upper_threshold,
                                                                                       last_inhibition)

            # ensures that only one output neuron can fire at a time
            out_spikes = (out_spike_locations * spike_occurred).to(dtype=inputs.dtype)

        with Timer('rate_track'):
            log_rates = log_rates_wo_inhibition - inhibition[..., None]

            # collect rates for plotting
            self.rate_tracker.update(relative_input_rates.clone().detach(), log_rates.clone().detach())

        if self.background_oscillation_active:
            states = (inhibition, noise[..., 0], phase)
        else:
            states = (inhibition, noise[..., 0])

        return out_spikes, states

    def get_spike_occurrences_and_inhibition(self, inhibition_upper_threshold, last_inhibition):
        inhibition = torch.zeros(inhibition_upper_threshold.shape[0] + 1,
                                 dtype=inhibition_upper_threshold.dtype,
                                 device=inhibition_upper_threshold.device)
        inhibition[0] = last_inhibition
        spike_occurred = torch.zeros_like(inhibition_upper_threshold, dtype=torch.bool)

        for ts in range(inhibition_upper_threshold.shape[0]):
            spike_occurred[ts] = inhibition[ts] < inhibition_upper_threshold[ts]
            inhibition[ts + 1] = (
                    self.inhibition_rest + (inhibition[ts] - self.inhibition_rest) * self.inhibition_factor
                    + spike_occurred[ts] * self.inhibition_increase)

        return inhibition[1:], spike_occurred


class EfficientBayesianSTDPModel(nn.Module):
    """Variant of BayesianSTDPModel that performs STDP updates and inference in a more efficient way.
        As STDP updates are performed in batches, this might lead to a slight difference in the results.
    """

    def __init__(self, input_neuron_cnt, output_neuron_cnt,
                 input_psp, multi_step_output_neuron_cell: EfficientStochasticOutputNeuronCell,
                 stdp_module,
                 track_states=False):
        super().__init__()
        self.linear = nn.Linear(input_neuron_cnt, output_neuron_cnt, bias=True).requires_grad_(False)
        self.output_neuron_cell = multi_step_output_neuron_cell

        self.input_psp = input_psp
        self.inhibition_tracker = InhibitionStateTracker(is_active=track_states, is_batched=True)
        self.weight_tracker = WeightsTracker(is_active=track_states)

        self.stdp_module = stdp_module

    @measure_time
    def forward(self, input_spikes: Tensor, state=None, train: bool = True) \
            -> (Tensor, Tensor):
        with torch.no_grad():
            z_state = state

            # (time, batch, input_neurons)
            input_psps = self.input_psp(input_spikes)

            z_in = self.linear(input_psps)
            z_out, z_states = self.output_neuron_cell(z_in, z_state)

            with Timer('state_metric'):
                # collect inhibition/noise states for plotting
                self.inhibition_tracker.update((z_states[0], z_states[1]))

            if train:
                with Timer('stdp'):
                    new_weights, new_biases = self.stdp_module(input_psps, z_out,
                                                               self.linear.weight.data,
                                                               self.linear.bias.data)
                    self.linear.weight.data = new_weights
                    self.linear.bias.data = new_biases

                    # collect weights for plotting
                    self.weight_tracker.update(new_weights.clone().detach(), new_biases.clone().detach())

            last_state = [state[-1] for state in z_states]

            return z_out, last_state
