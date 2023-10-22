import numpy as np


def spike_in_range(spikes, time_ranges):
    in_range = np.zeros(spikes.shape, dtype=bool)
    for (range_start, range_end) in time_ranges:
        in_range |= ((spikes <= range_end) & (spikes >= range_start))

    return in_range
