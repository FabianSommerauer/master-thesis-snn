import torch
from torchmetrics import metric
import matplotlib.pyplot as plt


class InhibitionStateTracker(metric.Metric):
    """Collects the inhibition/noise states of a neuron in a list."""

    def __init__(self, is_active=True, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.is_active = is_active
        self.add_state("inhibition_states", default=[], dist_reduce_fx="cat")
        self.add_state("noise_states", default=[], dist_reduce_fx="cat")

    def update(self, state):
        if not self.is_active:
            return

        inhibition_states, noise_states = state
        self.inhibition_states.append(inhibition_states)
        self.noise_states.append(noise_states)

    def compute(self):
        return self.inhibition_states, self.noise_states

    def plot(self):
        """Plot total inhibition and noise (sum of both) over time."""
        inhibition_states, noise_states = self.compute()
        inhibition = torch.stack(inhibition_states, dim=0).cpu().numpy()
        noise = torch.stack(noise_states, dim=0).cpu().numpy()
        # plt.plot(inhibition, label='Inhibition')
        # plt.plot(noise, label='Noise')
        plt.plot(inhibition + noise, label='Total')
        # plt.legend()
        plt.show()


class SpikeRateTracker(metric.Metric):
    """Collects the spike rates of each output neuron over time."""

    def __init__(self, is_active=True, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.is_active = is_active
        self.add_state("input_spike_rates", default=[], dist_reduce_fx="cat")
        self.add_state("log_total_spike_rates", default=[], dist_reduce_fx="cat")

    def update(self, input_rates, log_total_spike_rates):
        if not self.is_active:
            return

        self.input_spike_rates.append(input_rates)
        self.log_total_spike_rates.append(log_total_spike_rates)

    def compute(self):
        if not self.is_active:
            return None, None

        input_rates = torch.stack(self.input_spike_rates, dim=0)
        log_total_spike_rates = torch.stack(self.log_total_spike_rates, dim=0)

        normalized_rates = input_rates / torch.sum(input_rates, dim=-1, keepdim=True)
        return normalized_rates, log_total_spike_rates

    def plot(self):
        """Plot relative firing rates and log total firing rates over time."""
        normalized_rates, log_total_spike_rates = self.compute()
        normalized_rates = normalized_rates.cpu().numpy()
        log_total_spike_rates = log_total_spike_rates.cpu().numpy()

        for i in range(normalized_rates.shape[-1]):
            plt.plot(normalized_rates[:, i], label=f'Rate Output neuron {i}')
        plt.title('Relative Firing Rates')
        plt.xlabel('Time Step')
        plt.ylabel('Rate')
        plt.legend()
        plt.show()

        for i in range(log_total_spike_rates.shape[-1]):
            plt.plot(log_total_spike_rates[:, i], label=f'Log Total Rate Output neuron {i}')
        plt.title('Log Total Firing Rates')
        plt.xlabel('Time Step')
        plt.ylabel('Rate')
        plt.legend()
        plt.show()

    def plot_relative_firing_rates(self, ax, colors=None):
        """Plot input rates over time."""
        rel_firing_rates, _ = self.compute()
        rel_firing_rates = rel_firing_rates.cpu().numpy()

        for i in range(rel_firing_rates.shape[-1]):
            if colors is None:
                ax.plot(rel_firing_rates[:, i], label=f'Output neuron {i}')
            else:
                ax.plot(rel_firing_rates[:, i], label=f'Output neuron {i}', color=colors[i])
        ax.set_title('Relative Firing Rates')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Rate')
        ax.legend()


class LearningRatesTracker(metric.Metric):
    """Collects the learning rates of each output neuron over time."""

    def __init__(self, is_active=True, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.is_active = is_active
        self.add_state("mu_w_history", default=[], dist_reduce_fx="cat")
        self.add_state("mu_b_history", default=[], dist_reduce_fx="cat")

    def update(self, mu_w, mu_b):
        if not self.is_active:
            return

        self.mu_w_history.append(mu_w)
        self.mu_b_history.append(mu_b)

    def compute(self):
        return torch.stack(self.mu_w_history), torch.stack(self.mu_b_history)

    def plot(self):
        mu_w_history, mu_b_history = self.compute()

        # todo: this has different interpretations for different STDP modules -> fix
        plt.plot(torch.mean(mu_w_history, -1)[:, 0], label='mu_w 0')
        plt.plot(torch.mean(mu_w_history, -1)[:, 1], label='mu_w 1')
        plt.plot(torch.mean(mu_w_history, -1)[:, 2], label='mu_w 2')
        plt.plot(torch.mean(mu_b_history, -1)[:], label='mu_b')
        plt.yscale('log')
        plt.legend()
        plt.show()


class WeightsTracker(metric.Metric):
    """Collects the weights and biases of a layer over time."""

    def __init__(self, is_active=True, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.is_active = is_active
        self.add_state("weights_history", default=[], dist_reduce_fx="cat")
        self.add_state("bias_history", default=[], dist_reduce_fx="cat")

    def update(self, weights, biases):
        if not self.is_active:
            return

        self.weights_history.append(weights)
        self.bias_history.append(biases)

    def compute(self):
        return torch.stack(self.weights_history), torch.stack(self.bias_history)

    def plot_biases_exp(self):
        weights_history, bias_history = self.compute()

        bias_exp = torch.exp(bias_history)

        for i in range(bias_exp.shape[-1]):
            plt.plot(bias_exp[:, i], label=f'bias {i}')
        plt.legend()
        plt.show()