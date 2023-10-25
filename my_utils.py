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


def get_predictions(output_spikes, time_ranges, pattern_mapping):
    """Get predictions by finding most responsive output neuron in each time range"""

    time_range_count = len(time_ranges)
    output_neuron_num = output_spikes.shape[-1]

    spike_counts = np.zeros((output_neuron_num,))

    predictions = np.ones((time_range_count,)) * -1.

    for range_idx, (range_start, range_end) in enumerate(time_ranges):
        for neuron_index in range(output_neuron_num):
            neuron_spikes = output_spikes[:, neuron_index]
            spike_times = np.where(neuron_spikes > 0)[0]

            spike_counts[neuron_index] = np.sum((spike_times <= range_end) & (spike_times >= range_start))

        if np.sum(spike_counts) == 0:
            predictions[range_idx] = -1.

        max_neuron = np.argmax(spike_counts)
        predictions[range_idx] = pattern_mapping[max_neuron]

    return predictions


def get_joint_probabilities_over_time(output_spikes, time_ranges_per_pattern):
    """Get joint probabilities of output neuron firing and pattern presentation over time"""

    total_time = output_spikes.shape[0]
    output_neuron_num = output_spikes.shape[-1]
    pattern_num = len(time_ranges_per_pattern)

    counts_over_time = np.zeros((total_time, pattern_num, output_neuron_num))

    spike_times = np.where(output_spikes > 0)

    for pattern_idx, time_ranges in enumerate(time_ranges_per_pattern):
        time_range = np.stack(time_ranges, axis=0)
        in_range = np.any((spike_times[0][:, None] <= time_range[:, 1]) & (spike_times[0][:, None] >= time_range[:, 0]),
                          axis=-1)

        counts_over_time[spike_times[0][in_range], pattern_idx, spike_times[-1][in_range]] = 1.

    cumulative_counts = np.cumsum(counts_over_time, axis=0)

    # avoid division by zero (assume "worst case" where no pattern is presented)
    cumulative_counts[cumulative_counts == 0] = 1.

    totals = np.sum(np.sum(cumulative_counts, axis=-1, keepdims=True), axis=-2, keepdims=True)

    probabilities = cumulative_counts / totals

    return probabilities


# ensure documentation mentions shapes of numpy arguments

def normalized_conditional_cross_entropy_paper(joint_probabilities):
    """Compute normalized conditional cross entropy H(k|Z) / H(k,Z) based on joint probabilities
    This is the normalization scheme described in the paper (https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003037).
    However, this metric is more difficult to interpret as the maximum value is not 1.0.

    Args:
        joint_probabilities: joint probabilities of pattern presentation and output neuron firing [shape (time, pattern, neuron) or (pattern, neuron)]
    """

    # compute conditional probabilities
    conditional_probabilities = joint_probabilities / np.sum(joint_probabilities, axis=-1, keepdims=True)

    # avoid division by zero (assume "worst case" where no pattern is presented)
    conditional_probabilities[conditional_probabilities == 0] = 1.  # todo: check if this is correct

    # compute cross entropy
    cross_entropy = -np.sum(np.sum(joint_probabilities * np.log(joint_probabilities), axis=-1), axis=-2)
    conditional_cross_entropy = -np.sum(np.sum(joint_probabilities * np.log(conditional_probabilities), axis=-1),
                                        axis=-2)

    # normalize
    normalized_cond_cross_entropy = conditional_cross_entropy / cross_entropy

    return normalized_cond_cross_entropy


def normalized_conditional_cross_entropy(joint_probabilities):
    """Compute normalized conditional cross entropy H(k|Z) / H(k) based on joint probabilities
    We use a different normalization scheme than described in the paper (https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003037).
    In our case the normalized value is always within [0,1]. 1 if k and Z are independent, 0 if Z perfectly predicts k.

    Args:
        joint_probabilities: joint probabilities of pattern presentation and output neuron firing [shape (time, pattern, neuron) or (pattern, neuron)]
    """

    # compute conditional probabilities
    conditional_probabilities = joint_probabilities / np.sum(joint_probabilities, axis=-1, keepdims=True)

    # avoid division by zero (assume "worst case" where no pattern is presented)
    conditional_probabilities[conditional_probabilities == 0] = 1.  # todo: check if this is correct

    pattern_probabilities = np.sum(joint_probabilities, axis=-1)

    # compute cross entropy
    conditional_cross_entropy = -np.sum(np.sum(joint_probabilities * np.log(conditional_probabilities), axis=-1),
                                        axis=-1)
    pattern_cross_entropy = -np.sum(pattern_probabilities * np.log(pattern_probabilities), axis=-1)

    # normalize
    normalized_cond_cross_entropy = conditional_cross_entropy / pattern_cross_entropy

    return normalized_cond_cross_entropy
