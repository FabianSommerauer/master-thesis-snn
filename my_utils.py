import numpy as np


def spike_in_range(spike_times, time_ranges):
    in_range = np.zeros(spike_times.shape, dtype=bool)
    for (range_start, range_end) in time_ranges:
        in_range |= ((spike_times <= range_end) & (spike_times >= range_start))

    return in_range


# TODO: this is not efficient
# TODO: this allows multiple neurons to be mapped to the same pattern
def get_neuron_pattern_mapping(output_spikes, pattern_time_ranges):
    """Map each output neuron to the pattern that it fires most frequently to."""
    neuron_pattern_mapping = []
    pattern_num = len(pattern_time_ranges)
    output_neuron_num = output_spikes.shape[-1]

    spike_counts = np.zeros((pattern_num,))

    for neuron_index in range(output_neuron_num):
        neuron_spikes = output_spikes[:, neuron_index]
        spike_times = np.where(neuron_spikes > 0)[0]

        for pattern_index, pattern_time_range in enumerate(pattern_time_ranges):
            spike_counts[pattern_index] = np.sum(spike_in_range(spike_times, pattern_time_range))

        neuron_pattern_mapping.append(np.argmax(spike_counts))

    return neuron_pattern_mapping
