from typing import Literal, Tuple
import torch
import torch.nn as nn
from my_metrics import LearningRatesTracker
from my_timing_utils import Timer


class BayesianSTDPClassic(nn.Module):
    def __init__(self, output_size, c: float = 1, base_mu: float = 1, base_mu_bias: float = 1,
                 collect_history: bool = False):
        super().__init__()
        self.base_mu = base_mu
        self.base_mu_bias = base_mu_bias
        self.output_size = output_size

        self.N_k = torch.ones((output_size, 1))

        self.c = c

        self.learning_rates_tracker = LearningRatesTracker(is_active=collect_history)

    def reset(self):
        self.N_k = torch.ones((self.output_size, 1))
        self.learning_rates_tracker.reset()  # todo: should this be here?

    def forward(self, input_psp: torch.Tensor, output_spikes: torch.Tensor,
                weights: torch.Tensor, biases: torch.Tensor):
        with torch.no_grad():
            with Timer('mu_update'):
                mu_w = self.base_mu / self.N_k
                mu_b = self.base_mu_bias / torch.sum(self.N_k, dim=0)

            with Timer('apply_bayesian_stdp'):
                new_weights, new_biases = apply_bayesian_stdp(input_psp, output_spikes, weights, biases,
                                                              mu_w, mu_b, self.c)

            with Timer('counter_update'):
                # (output_count)
                out_dims = output_spikes.dim()
                if out_dims == 1:
                    total_out_spikes = output_spikes
                else:
                    total_out_spikes = torch.sum(output_spikes, tuple(range(out_dims - 1)))

                self.N_k += total_out_spikes[:, None]

            # with Timer('track_learning_rates'):
            #     # collect learning rates for plotting
            #     self.learning_rates_tracker(mu_w, mu_b)
            #     pass

            return new_weights, new_biases


class BayesianSTDPAdaptive(nn.Module):
    def __init__(self, input_size, output_size, c: float = 1, collect_history: bool = False):
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
                                                          self.mu_w, self.mu_b, self.c)

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

                self.bias_first_moment = self.mu_b * biases + (1 - self.mu_b) * self.bias_first_moment
                self.bias_second_moment = self.mu_b * biases ** 2 + (1 - self.mu_b) * self.bias_second_moment

            # adaptive learning rates with variance tracking
            self.mu_w = (self.weight_second_moment - self.weight_first_moment ** 2) / (
                    torch.exp(-self.weight_first_moment) + 1.0)
            self.mu_b = (self.bias_second_moment - self.bias_first_moment ** 2) / (
                    torch.exp(-self.bias_first_moment) + 1.0)

            # collect learning rates for plotting
            self.learning_rates_tracker(self.mu_w, self.mu_b)

            return new_weights, new_biases


def apply_bayesian_stdp(
        input_psp: torch.Tensor,
        output_spikes: torch.Tensor,
        weights: torch.Tensor,
        biases: torch.Tensor,
        mu_weights: torch.Tensor,
        mu_bias: torch.Tensor,
        c: float = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """STDP step for bayesian computation.
    Input:
        input_psp (torch.Tensor): Postsynaptic potential induced by each input neuron during single time step; shape: (batch, input_count)
        output_spikes (torch.Tensor): Spikes of the output neurons at single time step; shape: (batch, output_count)
        weights (torch.Tensor): Weight tensor of linear layer connecting pre- and postsynaptic layers; shape: (output_count, input_count)
        biases (torch.Tensor): Bias tensor of linear layer connecting pre- and postsynaptic layers; shape: (output_count)
        mu_weights (torch.Tensor): learning rates for the weights
        mu_bias (torch.Tensor): learning rates for the bias
        c (float): constant determining shift of final weights
    Output:
        weights (torch.tensor): Updated weights
        biases (torch.tensor): Updated biases
    """

    # (output_count)
    out_dims = output_spikes.dim()
    total_out_spikes = output_spikes if out_dims == 1 \
        else output_spikes.sum(dim=tuple(range(out_dims - 1)))

    with Timer('dw_db'):
        # STDP weight update
        # only applies to active neuron
        dw = torch.einsum('...o,...i->oi', output_spikes, input_psp) * c * torch.exp(-weights) - total_out_spikes[:, None]
        # dw = torch.matmul(output_spikes[..., None], input_psp[..., None, :]) * c * torch.exp(-weights) - total_out_spikes[:, None]

        # applies to all neurons (if at least one fired)
        db = (torch.exp(-biases) * total_out_spikes - 1) * total_out_spikes.any()

    weights = weights + mu_weights * dw
    biases = biases + mu_bias * db

    return weights, biases
