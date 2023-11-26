import numpy as np
import torch
from matplotlib import ticker, pyplot as plt

from my_utils import spike_in_range


# Define a function to create a raster plot for spikes
def raster_plot(ax, spikes, color='b'):
    for i, spike_train in enumerate(spikes):
        ax.eventplot(np.where(spike_train > 0)[0], lineoffsets=i, linelengths=0.7, colors=color)


def eventplot_time_range_colored(ax, spike_coords, time_ranges_per_color, colors, default_color='black',
                                 lineoffsets=0.0, linelengths=0.7):
    in_any_range = np.zeros(spike_coords.shape, dtype=bool)
    for color, time_ranges in zip(colors, time_ranges_per_color):
        in_range = spike_in_range(spike_coords, time_ranges)
        in_any_range |= in_range
        ax.eventplot(spike_coords[in_range], lineoffsets=lineoffsets, linelengths=linelengths, colors=color)

    ax.eventplot(spike_coords[~in_any_range], lineoffsets=lineoffsets, linelengths=linelengths, colors=default_color)


def raster_plot_multi_color(ax, spikes, time_ranges_per_color, colors, default_color='black',
                            allowed_colors_per_train=None):
    for i, spike_train in enumerate(spikes):
        spike_coords = np.where(spike_train > 0)[0]

        if allowed_colors_per_train is not None:
            allowed_color_indices = allowed_colors_per_train[i]
            time_ranges_per_allowed_color = [time_ranges_per_color[i] for i in allowed_color_indices]
            allowed_colors = [colors[i] for i in allowed_color_indices]
            eventplot_time_range_colored(ax, spike_coords,
                                         time_ranges_per_allowed_color, allowed_colors,
                                         default_color=default_color,
                                         lineoffsets=i, linelengths=0.7)
        else:
            eventplot_time_range_colored(ax, spike_coords, time_ranges_per_color, colors, default_color=default_color,
                                         lineoffsets=i, linelengths=0.7)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))


def raster_plot_multi_color_per_train(ax, spikes, time_ranges_per_color_per_train, colors, default_color='black'):
    for i, spike_train in enumerate(spikes):
        spike_coords = np.where(spike_train > 0)[0]

        time_ranges_per_color = time_ranges_per_color_per_train[i]

        eventplot_time_range_colored(ax, spike_coords, time_ranges_per_color, colors, default_color=default_color,
                                     lineoffsets=i, linelengths=0.7)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))


def plot_weight_visualization(weights, grid_size, image_size, stride=2, offset=0, exp=True, save_path=None):
    width, height = image_size
    grid_width, grid_height = grid_size

    width, height, grid_width, grid_height = int(width), int(height), int(grid_width), int(grid_height)

    if exp:
        weights = torch.exp(weights)
    weights = weights.cpu().numpy()

    output_neuron_count = weights.shape[0]

    fig, axs = plt.subplots(grid_height, grid_width)
    for i in range(grid_height):
        for j in range(grid_width):
            ax = axs[i, j]
            ax.axis('off')

            neuron_idx = i * grid_width + j
            if neuron_idx >= output_neuron_count:
                continue

            neuron_weights = weights[neuron_idx, offset::stride]
            if neuron_weights.shape[0] < width * height:
                neuron_weights = np.pad(neuron_weights, (0, width * height - neuron_weights.shape[0]))

            ax.imshow(neuron_weights.reshape(height, width))

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
