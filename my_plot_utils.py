import numpy as np
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


def raster_plot_multi_color_per_train(ax, spikes, time_ranges_per_color_per_train, colors, default_color='black'):
    for i, spike_train in enumerate(spikes):
        spike_coords = np.where(spike_train > 0)[0]

        time_ranges_per_color = time_ranges_per_color_per_train[i]

        eventplot_time_range_colored(ax, spike_coords, time_ranges_per_color, colors, default_color=default_color,
                                     lineoffsets=i, linelengths=0.7)