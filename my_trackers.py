import numpy as np
import torch
from torchmetrics import metric
import matplotlib.pyplot as plt


class InhibitionStateTracker:
    """Collects the inhibition/noise states of a neuron in a list."""

    def __init__(self, is_active=True, is_batched=False):
        self.is_active = is_active
        self.inhibition_states = []
        self.noise_states = []
        self.is_batched = is_batched

    def update(self, state):
        if not self.is_active:
            return

        inhibition_states, noise_states = state
        self.inhibition_states.append(inhibition_states)
        self.noise_states.append(noise_states)

    def compute(self):
        if len(self.inhibition_states) == 0 or len(self.noise_states) == 0:
            return None, None

        if self.is_batched:
            inhibition_states = torch.concat(self.inhibition_states, dim=0)
            noise_states = torch.concat(self.noise_states, dim=0)
        else:
            inhibition_states = torch.stack(self.inhibition_states, dim=0)
            noise_states = torch.stack(self.noise_states, dim=0)

        return inhibition_states, noise_states

    def reset(self):
        self.inhibition_states = []
        self.noise_states = []

    def plot(self):
        """Plot total inhibition and noise (sum of both) over time."""
        inhibition_states, noise_states = self.compute()

        if inhibition_states is None or noise_states is None:
            return

        inhibition = inhibition_states.cpu().numpy()
        noise = noise_states.cpu().numpy()

        # plt.plot(inhibition, label='Inhibition')
        # plt.plot(noise, label='Noise')
        plt.plot(inhibition + noise, label='Total')
        # plt.legend()
        plt.show()


class SpikeRateTracker:
    """Collects the spike rates of each output neuron over time."""

    def __init__(self, is_active=True, is_batched=False):
        self.is_active = is_active
        self.input_spike_rates = []
        self.log_total_spike_rates = []
        self.is_batched = is_batched

    def update(self, input_rates, log_total_spike_rates):
        if not self.is_active:
            return

        self.input_spike_rates.append(input_rates)
        self.log_total_spike_rates.append(log_total_spike_rates)

    def compute(self):
        if not self.is_active:
            return None, None

        if len(self.input_spike_rates) == 0 or len(self.log_total_spike_rates) == 0:
            return None, None

        if self.is_batched:
            input_rates = torch.concat(self.input_spike_rates, dim=0)
            log_total_spike_rates = torch.concat(self.log_total_spike_rates, dim=0)
        else:
            input_rates = torch.stack(self.input_spike_rates, dim=0)
            log_total_spike_rates = torch.stack(self.log_total_spike_rates, dim=0)

        normalized_rates = input_rates / torch.sum(input_rates, dim=-1, keepdim=True)
        return normalized_rates, log_total_spike_rates

    def reset(self):
        self.input_spike_rates = []
        self.log_total_spike_rates = []

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


class LearningRatesTracker:
    """Collects the learning rates of each output neuron over time."""

    def __init__(self, is_active=True):
        self.is_active = is_active
        self.mu_w_history = []
        self.mu_b_history = []

    def update(self, mu_w, mu_b):
        if not self.is_active:
            return

        self.mu_w_history.append(mu_w)
        self.mu_b_history.append(mu_b)

    def compute(self):
        return torch.stack(self.mu_w_history), torch.stack(self.mu_b_history)

    def reset(self):
        self.mu_w_history = []
        self.mu_b_history = []

    def plot(self):
        mu_w_history, mu_b_history = self.compute()

        # # todo: this has different interpretations for different STDP modules -> fix
        # plt.plot(torch.mean(mu_w_history, -1)[:, 0], label='mu_w 0')
        # plt.plot(torch.mean(mu_w_history, -1)[:, 1], label='mu_w 1')
        # plt.plot(torch.mean(mu_w_history, -1)[:, 2], label='mu_w 2')
        # plt.plot(torch.mean(mu_b_history, -1)[:], label='mu_b')

        mu_w_hist_mean = torch.mean(mu_w_history, -1)
        mu_b_hist_mean = torch.mean(mu_b_history, -1)

        for i in range(mu_w_hist_mean.shape[-1]):
            plt.plot(mu_w_hist_mean[:, i], label=f'mu_w {i}')
        plt.plot(mu_b_hist_mean, label='mu_b')

        plt.yscale('log')
        plt.legend()
        plt.show()


class WeightsTracker:
    """Collects the weights and biases of a layer over time."""

    def __init__(self, is_active=True):
        self.is_active = is_active
        self.weights_history = []
        self.bias_history = []

    def update(self, weights, biases):
        if not self.is_active:
            return

        self.weights_history.append(weights)
        self.bias_history.append(biases)

    def compute(self):
        if not self.is_active:
            return None, None

        if len(self.weights_history) == 0 or len(self.bias_history) == 0:
            return None, None

        return torch.stack(self.weights_history), torch.stack(self.bias_history)

    def reset(self):
        self.weights_history = []
        self.bias_history = []

    def plot_bias_convergence(self, target_biases=None, colors=None, exp=True):
        weights_history, bias_history = self.compute()

        if weights_history is None or bias_history is None:
            return

        if exp:
            bias_history = torch.exp(bias_history)
        bias_history = bias_history.cpu().numpy()

        for i in range(bias_history.shape[-1]):
            label = f'exp(bias_{i})' if exp else f'bias_{i}'
            plt.plot(bias_history[:, i], label=label, color=colors[i] if colors is not None else None)
        if target_biases is not None:
            for i in range(len(target_biases)):
                tgt = target_biases[i]
                if exp:
                    tgt = np.exp(tgt)
                plt.axhline(tgt, color=colors[i] if colors is not None else None, linestyle='--',
                            label=f'target bias {i}')

        plt.title('Bias convergence')
        plt.legend()
        plt.show()

    def plot_final_weight_visualization(self, grid_size, image_size, stride=2, offset=0, exp=True):
        width, height = image_size
        grid_width, grid_height = grid_size

        width, height, grid_width, grid_height = int(width), int(height), int(grid_width), int(grid_height)

        weights_history, bias_history = self.compute()

        weights = weights_history[-1]
        if exp:
            weights = torch.exp(weights)
        weights = weights.cpu().numpy()

        weight_count = weights.shape[1]

        fig, axs = plt.subplots(grid_height, grid_width)
        for i in range(grid_height):
            for j in range(grid_width):
                ax = axs[i, j]
                ax.axis('off')
                if i * grid_width + j >= weight_count:
                    continue

                ax.imshow(weights[i * grid_width + j, offset::stride].reshape(height, width))

        plt.show()