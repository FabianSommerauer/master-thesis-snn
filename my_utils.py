import random

import numpy as np
import torch
from einops import rearrange
from torch import Tensor


def spike_in_range(spike_times, time_ranges):
    in_range = np.zeros(spike_times.shape, dtype=bool)
    for (range_start, range_end) in time_ranges:
        in_range |= ((spike_times <= range_end) & (spike_times >= range_start))

    return in_range


# TODO: this is not efficient
# TODO: this allows multiple neurons to be mapped to the same pattern
# TODO: this could also be done using cumulative counts
def get_neuron_pattern_mapping(output_spikes, pattern_time_ranges):
    """Map each output neuron to the pattern that it fires most frequently to."""
    neuron_pattern_mapping = []
    pattern_num = len(pattern_time_ranges)
    output_neuron_num = output_spikes.shape[-1]

    for neuron_index in range(output_neuron_num):
        spike_counts = np.zeros((pattern_num,))
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

    predictions = np.ones((time_range_count,), dtype=int) * -1

    for range_idx, (range_start, range_end) in enumerate(time_ranges):
        for neuron_index in range(output_neuron_num):
            neuron_spikes = output_spikes[:, neuron_index]
            spike_times = np.where(neuron_spikes > 0)[0]

            if len(spike_times) == 0:
                continue

            spike_counts[neuron_index] = np.sum((spike_times <= range_end) & (spike_times >= range_start))

        if np.sum(spike_counts) == 0:
            predictions[range_idx] = -1.
            continue

        max_neuron = np.argmax(spike_counts)
        predictions[range_idx] = pattern_mapping[max_neuron]

    return predictions


def get_predictions_from_rates(output_rates, time_ranges, pattern_mapping):
    """Get predictions by finding most responsive output neuron in each time range
    Most responsive neuron is determined by comparing the averaged rates within each timeframe
    """
    time_range_count = len(time_ranges)
    time_step_count, output_neuron_num = output_rates.shape

    predictions = np.ones((time_range_count,)) * -1.

    for range_idx, (range_start, range_end) in enumerate(time_ranges):
        start_idx = max(round(range_start), 0)
        end_idx = min(round(range_end), time_step_count - 1)
        avg_rates = np.mean(output_rates[start_idx:end_idx], axis=0)

        max_neuron = np.argmax(avg_rates)
        predictions[range_idx] = pattern_mapping[max_neuron]

    return predictions


def get_cumulative_counts_over_time(output_spikes, time_ranges_per_pattern, base_counts=None, time_offset=0.0):
    """Get cumulative counts of output neuron firing over time

    Args:
        output_spikes: output spikes [shape (time, neuron)]
        time_ranges_per_pattern: time ranges for each pattern [shape (pattern, time_range, 2)] todo: currently this is just a list of lists
        base_counts: base counts to add to the counts for each pattern [shape (pattern, neuron)]
    """

    total_time = output_spikes.shape[0]
    output_neuron_num = output_spikes.shape[-1]
    pattern_num = len(time_ranges_per_pattern)

    counts_over_time = np.zeros((total_time, pattern_num, output_neuron_num))

    spike_times = np.where(output_spikes > 0)

    for pattern_idx, time_ranges in enumerate(time_ranges_per_pattern):
        if len(time_ranges) == 0:
            continue
        time_range = np.stack(time_ranges, axis=0)
        in_range = np.any((spike_times[0][:, None] <= time_range[:, 1] - time_offset) & (
                spike_times[0][:, None] >= time_range[:, 0] - time_offset),
                          axis=-1)

        counts_over_time[spike_times[0][in_range], pattern_idx, spike_times[-1][in_range]] = 1.

    cumulative_counts = np.cumsum(counts_over_time, axis=0)

    if base_counts is not None:
        cumulative_counts += base_counts[None, ...]

    return cumulative_counts


def get_joint_probabilities_from_counts(counts, epsilon=1):
    """Get joint probabilities of output neuron firing and pattern presentation from counts

    Args:
        counts: counts of output neuron firing and pattern presentation [shape (time, pattern, neuron)]
        epsilon: small constant to avoid division by zero and to prevent high joint probabilities due to low counts (default: 1)
    """

    # avoid division by zero (assume "worst case" where no pattern is presented)
    counts_cp = counts.copy()
    counts_cp[counts_cp == 0] = epsilon

    totals = np.sum(np.sum(counts_cp, axis=-1, keepdims=True), axis=-2, keepdims=True)

    probabilities = counts_cp / totals

    return probabilities


def get_joint_probabilities_over_time(output_spikes, time_ranges_per_pattern, base_counts=None):
    """Get joint probabilities of output neuron firing and pattern presentation over time

    Args:
        output_spikes: output spikes [shape (time, neuron)]
        time_ranges_per_pattern: time ranges for each pattern [shape (pattern, time_range, 2)] todo: currently this is just a list of lists
        base_counts: base counts to add to the counts for each pattern [shape (pattern, neuron)]
    """

    cumulative_counts = get_cumulative_counts_over_time(output_spikes, time_ranges_per_pattern, base_counts)
    probabilities = get_joint_probabilities_from_counts(cumulative_counts)
    return probabilities


# todo: check if this works correctly
def get_joint_probabilities_over_time_for_rates(relative_firing_rates, time_ranges_per_pattern):
    """Get joint probabilities of output neuron firing and pattern presentation over time

    Args:
        relative_firing_rates: relative firing rates [shape (time, neuron)]
        time_ranges_per_pattern: time ranges for each pattern [shape (pattern, time_range, 2)] todo: currently this is just a list of lists
    """

    total_time = relative_firing_rates.shape[0]
    output_neuron_num = relative_firing_rates.shape[-1]
    pattern_num = len(time_ranges_per_pattern)

    joint_probabilities_over_time = np.zeros((total_time, pattern_num, output_neuron_num))

    potential_spike_times = np.arange(total_time)

    for pattern_idx, time_ranges in enumerate(time_ranges_per_pattern):
        time_range = np.stack(time_ranges, axis=0)
        in_range = np.any(
            (potential_spike_times[:, None] <= time_range[:, 1]) & (potential_spike_times[:, None] >= time_range[:, 0]),
            axis=-1)

        joint_probabilities_over_time[potential_spike_times[in_range], pattern_idx, :] = relative_firing_rates[
                                                                                         potential_spike_times[
                                                                                             in_range], :]

    return joint_probabilities_over_time


def normalized_conditional_cross_entropy_paper(joint_probabilities):
    """Compute normalized conditional cross entropy H(k|Z) / H(k,Z) based on joint probabilities
    This is the normalization scheme described in the paper (https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003037).
    However, this metric is more difficult to interpret as the maximum value is not 1.0.

    Args:
        joint_probabilities: joint probabilities of pattern presentation and output neuron firing [shape (time, pattern, neuron) or (pattern, neuron)]
    """

    # todo: deal with zeros

    # compute conditional probabilities
    conditional_probabilities = joint_probabilities / np.sum(joint_probabilities, axis=-2, keepdims=True)

    # compute cross entropy
    cross_entropy = -np.sum(np.sum(joint_probabilities * np.log(joint_probabilities), axis=-1), axis=-1)
    conditional_cross_entropy = -np.sum(np.sum(joint_probabilities * np.log(conditional_probabilities), axis=-1),
                                        axis=-1)

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

    # todo: deal with zeros

    # compute conditional probabilities
    conditional_probabilities = joint_probabilities / np.sum(joint_probabilities, axis=-2, keepdims=True)

    pattern_probabilities = np.sum(joint_probabilities, axis=-1)

    # compute cross entropy
    conditional_cross_entropy = -np.sum(np.sum(joint_probabilities * np.log(conditional_probabilities), axis=-1),
                                        axis=-1)
    pattern_cross_entropy = -np.sum(pattern_probabilities * np.log(pattern_probabilities), axis=-1)

    # normalize
    normalized_cond_cross_entropy = conditional_cross_entropy / pattern_cross_entropy

    return normalized_cond_cross_entropy


def grouped_sum(array, groups):
    """Sum values in array based on group indices keeping the initial shape of the array

    Args:
        array: array to sum [shape arbitrary]
        groups: group indices [shape same as array]
    Output:
        grouped_sums: grouped sums [shape same as array]
    """

    grouped_sums = np.zeros(array.shape)

    unique_indices = np.unique(groups)

    for idx in unique_indices:
        mask = groups == idx
        grouped_sums[mask] = np.sum(array[mask])

    return grouped_sums


def get_input_likelihood(weights, biases, input_psp, input_groups, c=1.):
    """Compute log likelihood of input spikes given weights and biases (as weights and biases represent a learned distribution)

    Args:
        weights: weights of linear layer [shape (iteration, neuron, input)]
        biases: biases of linear layer [shape (iteration, neuron,)]
        input_psp: input psp; values assumed to be 0 or 1; should be grouped with each group always having exactly 1 active neuron [shape (iteration, time, input)]
        input_groups: group idx of each input (used to appropriately normalize weights) [shape (iteration, input,)]
        c: constant used to during learning (default: 1.)
    Output:
        total_input_likelihood: total likelihood of input spikes [shape (iteration, time)]
    """

    # todo: deal with input_psps where multiple neurons in each group may be active at once and which aren't binary

    iteration_count, neuron_count, input_count = weights.shape
    groups = np.repeat(np.repeat(input_groups[None, None, :], neuron_count, axis=1), iteration_count, axis=0)

    priors = np.exp(biases)
    normalized_priors = priors / np.sum(priors, axis=-1, keepdims=True)

    single_input_likelihoods = np.exp(weights) / c
    # todo: this assumes no duplicates in groups
    # todo: this will turn to nan if there are no spikes in a group -> fix
    group_normalized_single_input_log_likelihoods = np.log(single_input_likelihoods
                                                           / grouped_sum(single_input_likelihoods, groups))

    # input_log_likelihoods = np.sum(input_psp[:, :, None, :] * group_normalized_single_input_log_likelihoods[:, None, :, :],
    #                                axis=-1)
    input_log_likelihoods = np.einsum('itn,ion->ito', input_psp, group_normalized_single_input_log_likelihoods)

    total_input_likelihood = np.sum(np.exp(input_log_likelihoods) * normalized_priors[:, None, :], axis=-1)

    return total_input_likelihood


# todo
# joint = get_joint_probabilities_over_time_for_rates(probs, train_time_ranges)
# plt.plot(normalized_conditional_cross_entropy(moving_avg(joint, 500, 0)))
def moving_avg(arr, len, axis):
    filt = np.ones(len) / len
    return np.apply_along_axis(lambda m: np.convolve(m, filt, mode='valid'), axis=axis, arr=arr)


def reorder_dataset_by_targets(data, targets):
    data_per_target = []

    unique_targets = targets.unique()
    target_counts = torch.zeros(unique_targets.shape)

    for i in unique_targets:
        mask = (targets == i)
        data_per_target.append(data[mask])
        target_counts[i] = mask.sum()

    ordered_data = []
    ordered_targets = []
    for k in range(int(target_counts.max())):
        for i in range(unique_targets.shape[0]):
            if k < target_counts[i]:
                ordered_data.append(data_per_target[i][k])
                ordered_targets.append(unique_targets[i])

    ordered_data = torch.stack(ordered_data)
    ordered_targets = torch.stack(ordered_targets)

    return ordered_data, ordered_targets


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ToBinaryTransform(object):
    def __init__(self, thresh):
        self.thresh = thresh

    def __call__(self, data: Tensor):
        return (data > self.thresh).to(data.dtype)


class FlattenTransform(object):
    def __call__(self, data: Tensor):
        return torch.flatten(data)