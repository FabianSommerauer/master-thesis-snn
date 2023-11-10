from typing import Literal, Tuple
import torch
import torch.nn as nn
from my_metrics import LearningRatesTracker
from my_timing_utils import Timer


class BayesianSTDPClassic(nn.Module):
    def __init__(self, output_size, c: float = 1, base_mu: float = 1, base_mu_bias: float = 1,
                 time_batch_size: int = 10,
                 collect_history: bool = False):
        super().__init__()
        self.base_mu = base_mu
        self.base_mu_bias = base_mu_bias
        self.output_size = output_size

        self.N_k = torch.ones((output_size, 1))

        self.c = c

        self.time_batch_size = time_batch_size

        self.learning_rates_tracker = LearningRatesTracker(is_active=collect_history)

    def reset(self):
        self.N_k = torch.ones((self.output_size, 1))
        self.learning_rates_tracker.reset()  # todo: should this be here?

    def forward(self, input_psp: torch.Tensor, output_spikes: torch.Tensor,
                weights: torch.Tensor, biases: torch.Tensor):
        with (torch.no_grad()):
            with Timer('apply_bayesian_stdp'):
                new_weights, new_biases, self.N_k = \
                    apply_bayesian_stdp_with_learning_rate_update(input_psp, output_spikes,
                                                                  weights, biases,
                                                                  self.base_mu, self.base_mu_bias, self.N_k,
                                                                  self.c, self.time_batch_size)

            with Timer('track_learning_rates'):
                mu_w = self.base_mu / self.N_k
                mu_b = self.base_mu_bias / torch.sum(self.N_k, dim=0)

                # collect learning rates for plotting
                self.learning_rates_tracker.update(mu_w, mu_b)
                pass

            return new_weights, new_biases


class BayesianSTDPAdaptive(nn.Module):
    def __init__(self, input_size, output_size, c: float = 1,
                 time_batch_size: int = 10,
                 collect_history: bool = False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.mu_w = torch.ones((output_size, input_size))
        self.mu_b = torch.ones((output_size,))

        self.weight_first_moment = None
        self.weight_second_moment = None
        self.bias_first_moment = None
        self.bias_second_moment = None

        self.c = c

        self.time_batch_size = time_batch_size

        self.learning_rates_tracker = LearningRatesTracker(is_active=collect_history)
        # self.weight_first_moment_history = []
        # self.weight_second_moment_history = []
        # self.bias_first_moment_history = []
        # self.bias_second_moment_history = []

    def reset(self):
        self.mu_w = torch.ones((self.output_size, self.input_size))
        self.mu_b = torch.ones((self.output_size,))

        self.weight_first_moment = None
        self.weight_second_moment = None
        self.bias_first_moment = None
        self.bias_second_moment = None

        self.learning_rates_tracker.reset()  # todo: should this be here?

    def forward(self, input_psp: torch.Tensor, output_spikes: torch.Tensor,
                weights: torch.Tensor, biases: torch.Tensor):
        with torch.no_grad():
            new_weights, new_biases = apply_bayesian_stdp(input_psp, output_spikes, weights, biases,
                                                          self.mu_w, self.mu_b, self.c,
                                                          self.time_batch_size)

            if self.weight_first_moment is None:
                # start with variance of 1 to avoid getting stuck on initial parameter values
                self.weight_first_moment = torch.zeros_like(weights)
                self.weight_second_moment = torch.ones_like(weights)
                self.bias_first_moment = torch.zeros_like(biases)
                self.bias_second_moment = torch.ones_like(biases) / self.output_size
            else:
                # update moments and learning rates
                self.weight_first_moment = self.mu_w * new_weights + (1 - self.mu_w) * self.weight_first_moment
                self.weight_second_moment = self.mu_w * new_weights ** 2 + (1 - self.mu_w) * self.weight_second_moment

                # TODO
                # self.weight_first_moment = self.weight_first_moment / (1 - self.mu_w ** time_step + 1e-8)
                # self.weight_second_moment = self.weight_second_moment / (1 - self.mu_w ** time_step + 1e-8)

                self.bias_first_moment = self.mu_b * biases + (1 - self.mu_b) * self.bias_first_moment
                self.bias_second_moment = self.mu_b * biases ** 2 + (1 - self.mu_b) * self.bias_second_moment

                # TODO
                # self.bias_first_moment = self.bias_first_moment / (1 - self.mu_b ** time_step + 1e-8)
                # self.bias_second_moment = self.bias_second_moment / (1 - self.mu_b ** time_step + 1e-8)

            # adaptive learning rates with variance tracking
            self.mu_w = (self.weight_second_moment - self.weight_first_moment ** 2) / (
                    torch.exp(-self.weight_first_moment) + 1.0)
            self.mu_b = (self.bias_second_moment - self.bias_first_moment ** 2) / (
                    torch.exp(-self.bias_first_moment) + 1.0)

            # collect learning rates for plotting
            self.learning_rates_tracker.update(self.mu_w, self.mu_b)

            return new_weights, new_biases


@torch.jit.script
def apply_bayesian_stdp(
        input_psp: torch.Tensor,
        output_spikes: torch.Tensor,
        weights: torch.Tensor,
        biases: torch.Tensor,
        mu_weights: torch.Tensor,
        mu_bias: torch.Tensor,
        c: float = 1,
        time_batch_size: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """STDP step for bayesian computation.
    Input:
        input_psp (torch.Tensor): Postsynaptic potential induced by each input neuron during single time step; shape: (time, ..., input_count)
        output_spikes (torch.Tensor): Spikes of the output neurons at single time step; shape: (time, ..., output_count)
        weights (torch.Tensor): Weight tensor of linear layer connecting pre- and postsynaptic layers; shape: (output_count, input_count)
        biases (torch.Tensor): Bias tensor of linear layer connecting pre- and postsynaptic layers; shape: (output_count)
        mu_weights (torch.Tensor): learning rates for the weights
        mu_bias (torch.Tensor): learning rates for the bias
        c (float): constant determining shift of final weights
        time_batch_size (int): number of time steps to process at once (should not contain too many distinct output spikes)
    Output:
        weights (torch.tensor): Updated weights
        biases (torch.tensor): Updated biases
    """

    time_steps = input_psp.shape[0]

    if time_batch_size == 1:
        iterations = time_steps

        input_psps_batched = input_psp
        output_spikes_batched = output_spikes
    else:
        iterations = time_steps // time_batch_size

        input_psps_batched = torch.stack(input_psp.chunk(iterations, dim=0), dim=0)
        output_spikes_batched = torch.stack(output_spikes.chunk(iterations, dim=0), dim=0)

    out_dims = output_spikes_batched.dim()

    # (iterations, ..., output_count)
    total_out_spikes = output_spikes_batched

    for i in range(1, out_dims - 1):
        total_out_spikes = output_spikes_batched.sum(dim=i)

    # STDP weight update

    spike_correlations = torch.einsum('t...o,t...i->toi', output_spikes_batched, input_psps_batched)
    total_out_spikes_sum = total_out_spikes.sum(dim=1)  # Precompute any() result

    for i in range(iterations):
        # only applies to active neuron
        dw = spike_correlations[i] * c * torch.exp(-weights) - total_out_spikes[i, :, None]

        # applies to all neurons (if at least one fired)
        db = (torch.exp(-biases) * total_out_spikes[i] - 1) * total_out_spikes_sum[i]

        weights += mu_weights * dw
        biases += mu_bias * db

    return weights, biases


@torch.jit.script
def apply_bayesian_stdp_with_learning_rate_update(
        input_psp: torch.Tensor,
        output_spikes: torch.Tensor,
        weights: torch.Tensor,
        biases: torch.Tensor,
        base_mu_weights: float,
        base_mu_bias: float,
        N_k: torch.Tensor,
        c: float = 1,
        time_batch_size: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """STDP step for bayesian computation with adaptive learning rates.
    Allows for more accurately processing large batches of time steps.
    Input:
        input_psp (torch.Tensor): Postsynaptic potential induced by each input neuron during single time step; shape: (time, ..., input_count)
        output_spikes (torch.Tensor): Spikes of the output neurons at single time step; shape: (time, ..., output_count)
        weights (torch.Tensor): Weight tensor of linear layer connecting pre- and postsynaptic layers; shape: (output_count, input_count)
        biases (torch.Tensor): Bias tensor of linear layer connecting pre- and postsynaptic layers; shape: (output_count)
        base_mu_weights (torch.Tensor): base learning rates for the weights
        base_mu_bias (torch.Tensor): base learning rates for the bias
        N_k (torch.Tensor): number of spikes per output neuron
        c (float): constant determining shift of final weights
        time_batch_size (int): number of time steps to process at once (should not contain too many distinct output spikes)
    Output:
        weights (torch.tensor): Updated weights
        biases (torch.tensor): Updated biases
    """

    time_steps = input_psp.shape[0]

    if time_batch_size == 1:
        iterations = time_steps

        input_psps_batched = input_psp
        output_spikes_batched = output_spikes
    else:
        iterations = time_steps // time_batch_size

        input_psps_batched = torch.stack(input_psp.chunk(iterations, dim=0), dim=0)
        output_spikes_batched = torch.stack(output_spikes.chunk(iterations, dim=0), dim=0)

    out_dims = output_spikes_batched.dim()

    # (iterations, ..., output_count)
    total_out_spikes = output_spikes_batched

    for i in range(1, out_dims - 1):
        total_out_spikes = output_spikes_batched.sum(dim=i)

    # STDP weight update

    spike_correlations = torch.einsum('t...o,t...i->toi', output_spikes_batched, input_psps_batched)
    total_out_spikes_sum = total_out_spikes.sum(dim=1)  # Precompute any() result

    for i in range(iterations):
        mu_weights = base_mu_weights / N_k
        mu_bias = base_mu_bias / torch.sum(N_k, dim=0)

        # only applies to active neuron
        dw = spike_correlations[i] * c * torch.exp(-weights) - total_out_spikes[i, :, None]

        # applies to all neurons (if at least one fired)
        db = (torch.exp(-biases) * total_out_spikes[i] - 1) * total_out_spikes_sum[i]

        weights += mu_weights * dw
        biases += mu_bias * db

        # update counter
        N_k += total_out_spikes[i, :, None]

    return weights, biases, N_k


@torch.jit.script
def apply_bayesian_stdp_with_adaptive_learning_rate_update(
        input_psp: torch.Tensor,
        output_spikes: torch.Tensor,
        weights: torch.Tensor,
        biases: torch.Tensor,
        base_mu_weights: float,
        base_mu_bias: float,
        N_k: torch.Tensor,
        c: float = 1,
        time_batch_size: int = 10,):
    """STDP step for bayesian computation with adaptive learning rates. Uses variance tracking.
    Allows for more accurately processing large batches of time steps."""
    time_steps = input_psp.shape[0]

    if time_batch_size == 1:
        iterations = time_steps

        input_psps_batched = input_psp
        output_spikes_batched = output_spikes
    else:
        iterations = time_steps // time_batch_size

        input_psps_batched = torch.stack(input_psp.chunk(iterations, dim=0), dim=0)
        output_spikes_batched = torch.stack(output_spikes.chunk(iterations, dim=0), dim=0)

    out_dims = output_spikes_batched.dim()

    # (iterations, ..., output_count)
    total_out_spikes = output_spikes_batched

    for i in range(1, out_dims - 1):
        total_out_spikes = output_spikes_batched.sum(dim=i)

    # STDP weight update

    spike_correlations = torch.einsum('t...o,t...i->toi', output_spikes_batched, input_psps_batched)
    total_out_spikes_sum = total_out_spikes.sum(dim=1)

    for i in range(iterations):
        # TODO REMOVE
        mu_weights = base_mu_weights / N_k
        mu_bias = base_mu_bias / torch.sum(N_k, dim=0)

        # only applies to active neuron
        dw = spike_correlations[i] * c * torch.exp(-weights) - total_out_spikes[i, :, None]

        # applies to all neurons (if at least one fired)
        db = (torch.exp(-biases) * total_out_spikes[i] - 1) * total_out_spikes_sum[i]

        weights += mu_weights * dw
        biases += mu_bias * db

        # update counter TODO REMOVE
        N_k += total_out_spikes[i, :, None]


        # TODO
        # update moments and learning rates
        weight_first_moment = mu_weights * weights + (1 - mu_weights) * weight_first_moment
        weight_second_moment = mu_weights * weights ** 2 + (1 - mu_weights) * weight_second_moment

        # TODO
        # weight_first_moment = weight_first_moment / (1 - mu_weights ** time_step + 1e-8)

        bias_first_moment = mu_bias * biases + (1 - mu_bias) * bias_first_moment
        bias_second_moment = mu_bias * biases ** 2 + (1 - mu_bias) * bias_second_moment

        # TODO
        # bias_first_moment = bias_first_moment / (1 - mu_bias ** time_step + 1e-8)

        # adaptive learning rates with variance tracking
        mu_weights = (weight_second_moment - weight_first_moment ** 2) / (torch.exp(-weight_first_moment) + 1.0)
        mu_bias = (bias_second_moment - bias_first_moment ** 2) / (torch.exp(-bias_first_moment) + 1.0)

    return weights, biases, N_k

