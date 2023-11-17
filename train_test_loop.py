from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch

import custom_stdp
from my_spike_modules import BinaryTimedPSP, EfficientBayesianSTDPModel, LogFiringRateCalculationMode, NoiseArgs, \
    InhibitionArgs, EfficientStochasticOutputNeuronCell, SpikePopulationGroupBatchToTimeEncoder, \
    BackgroundOscillationArgs
from my_timing_utils import Timer
from my_trackers import SpikeRateTracker, InhibitionStateTracker, LearningRatesTracker, WeightsTracker
from my_utils import normalized_conditional_cross_entropy_paper, normalized_conditional_cross_entropy, \
    get_joint_probabilities_from_counts, get_cumulative_counts_over_time, get_input_likelihood, get_predictions, \
    get_predictions_from_rates, get_neuron_pattern_mapping


@dataclass
class EncoderConfig:
    presentation_duration: float
    delay: float
    active_rate: float
    inactive_rate: float
    background_oscillation_args: BackgroundOscillationArgs | None = None


@dataclass
class STDPConfig:
    base_mu: float
    base_mu_bias: float
    c: float
    time_batch_size: int


@dataclass
class OutputCellConfig:
    inhibition_args: InhibitionArgs
    noise_args: NoiseArgs
    log_firing_rate_calc_mode: LogFiringRateCalculationMode
    background_oscillation_args: BackgroundOscillationArgs | None = None


@dataclass
class ModelConfig:
    dt: float
    input_neuron_count: int
    output_neuron_count: int
    sigma: float

    encoder_config: EncoderConfig
    stdp_config: STDPConfig
    output_cell_config: OutputCellConfig

    weight_init: float | None = None
    bias_init: float | None = None


@dataclass
class TrainConfig:
    model_config: ModelConfig
    num_epochs: int
    distinct_target_count: int
    print_interval: int | None = 10


@dataclass
class TestConfig:
    distinct_target_count: int
    model_config: ModelConfig
    trained_params: Tuple[torch.Tensor, torch.Tensor] | None = None
    neuron_pattern_mapping: Optional[np.array] = None,
    print_results: bool = True


@dataclass
class TrainResults:
    cross_entropy_hist: np.array
    cross_entropy_paper_hist: np.array
    input_likelihood_hist: np.array

    rate_tracker: SpikeRateTracker
    inhibition_tracker: InhibitionStateTracker
    learning_rates_tracker: LearningRatesTracker
    weights_tracker: WeightsTracker

    trained_params: Tuple[torch.Tensor, torch.Tensor]
    neuron_pattern_mapping: np.array

    timing_info: str


@dataclass
class TestResults:
    accuracy: float
    rate_accuracy: float
    miss_rate: float
    confusion_matrix: np.array

    cross_entropy: float
    cross_entropy_paper: float
    average_input_likelihood: float

    rate_tracker: SpikeRateTracker
    inhibition_tracker: InhibitionStateTracker

    total_input_spikes: np.array
    total_output_spikes: np.array
    total_time_ranges: list[list[float]]

    timing_info: str


def init_model(model_config: ModelConfig) -> Tuple[SpikePopulationGroupBatchToTimeEncoder, EfficientBayesianSTDPModel]:
    encoder_config = model_config.encoder_config
    stdp_config = model_config.stdp_config
    output_cell_config = model_config.output_cell_config
    sigma = model_config.sigma
    dt = model_config.dt

    encoder = SpikePopulationGroupBatchToTimeEncoder(encoder_config.presentation_duration,
                                                     encoder_config.active_rate, encoder_config.inactive_rate,
                                                     encoder_config.delay, dt,
                                                     background_oscillation_args=encoder_config.background_oscillation_args)

    stdp_module = custom_stdp.BayesianSTDPClassic(model_config.output_neuron_count, c=stdp_config.c,
                                                  base_mu=stdp_config.base_mu, base_mu_bias=stdp_config.base_mu_bias,
                                                  time_batch_size=stdp_config.time_batch_size,
                                                  collect_history=True)
    # stdp_module = custom_stdp.BayesianSTDPAdaptive(input_neurons, output_neurons,
    #                                                time_batch_size=stdp_time_batch_size,
    #                                                c=1, collect_history=True)  #todo: get this to work

    output_cell = EfficientStochasticOutputNeuronCell(inhibition_args=output_cell_config.inhibition_args,
                                                      noise_args=output_cell_config.noise_args,
                                                      log_firing_rate_calc_mode=output_cell_config.log_firing_rate_calc_mode,
                                                      background_oscillation_args=output_cell_config.background_oscillation_args,
                                                      dt=dt, collect_rates=True)

    model = EfficientBayesianSTDPModel(model_config.input_neuron_count, model_config.output_neuron_count,
                                       BinaryTimedPSP(sigma, dt),
                                       multi_step_output_neuron_cell=output_cell,
                                       stdp_module=stdp_module, track_states=True,
                                       return_psp=True)

    # Model initialization
    if model_config.weight_init is not None:
        model.linear.weight.data.fill_(model_config.weight_init)
    if model_config.bias_init is not None:
        model.linear.bias.data.fill_(model_config.bias_init)

    return encoder, model


def train_model(config: TrainConfig, data_loader):
    num_epochs = config.num_epochs
    model_config = config.model_config
    stdp_config = model_config.stdp_config

    # Model setup
    encoder, model = init_model(model_config)

    # Metric tracking
    cumulative_counts_hist = []
    input_likelihood_hist = []
    total_output_spikes = []
    total_time_ranges = [[] for _ in range(config.distinct_target_count)]

    # Used for input likelihood calculation
    input_groups = np.repeat(np.arange(model_config.input_neuron_count // 2), 2)

    # Training
    offset = 0
    osc_phase = None
    state = None
    Timer.reset()
    for epoch in range(num_epochs):
        for i, (data, targets) in enumerate(iter(data_loader)):
            with Timer('training loop'):
                input_spikes, osc_phase = encoder(data, osc_phase)

                time_ranges = encoder.get_time_ranges_for_patterns(targets,
                                                                   distinct_pattern_count=config.distinct_target_count,
                                                                   offset=offset)

                output_spikes, state, input_psp = model(input_spikes, state=state, train=True)

                with Timer('metric_processing'):
                    output_spikes_np = output_spikes.cpu().numpy()
                    time_offset = encoder.get_time_for_offset(offset)

                    total_output_spikes.append(output_spikes_np)
                    for idx, time_range in enumerate(time_ranges):
                        total_time_ranges[idx].extend(time_range)

                    with Timer('cumulative_counts'):
                        cumulative_counts = get_cumulative_counts_over_time(np.array(output_spikes_np),
                                                                            time_ranges,
                                                                            base_counts=None if i == 0 else
                                                                            cumulative_counts[-1],
                                                                            time_offset=time_offset)
                        cumulative_counts_hist.append(cumulative_counts)

                    with Timer('input_likelihood'):
                        weight_np = model.linear.weight.data.cpu().numpy()
                        bias_np = model.linear.bias.data.cpu().numpy()
                        input_psp_np = input_psp.cpu().numpy()

                        input_likelihood = get_input_likelihood(weight_np, bias_np,
                                                                input_psp_np, input_groups, stdp_config.c)
                        input_likelihood_hist.append(input_likelihood)

                    offset += data.shape[0]

                    if config.print_interval is not None:
                        with Timer('metric_printing'):
                            if i % config.print_interval == 0 or i == len(data_loader) - 1:
                                with Timer('cross_entropy'):
                                    joint_probs = get_joint_probabilities_from_counts(cumulative_counts[-1])

                                    cond_cross_entropy = normalized_conditional_cross_entropy(joint_probs)

                                    cond_cross_entropy_paper = normalized_conditional_cross_entropy_paper(joint_probs)

                                print(f"Epoch {epoch}, Iteration {i} \n"
                                      f"Train Loss: {cond_cross_entropy:.4f}; Paper Loss: {cond_cross_entropy_paper:.4f}")

    cumulative_counts_concat = np.concatenate(cumulative_counts_hist, axis=0)
    joint_probs = get_joint_probabilities_from_counts(cumulative_counts_concat)
    cond_cross_entropy = normalized_conditional_cross_entropy(joint_probs)
    cond_cross_entropy_paper = normalized_conditional_cross_entropy_paper(joint_probs)

    input_likelihood_concat = np.concatenate(input_likelihood_hist, axis=0)

    output_spikes_concat = np.concatenate(total_output_spikes, axis=0)

    neuron_pattern_mapping = get_neuron_pattern_mapping(output_spikes_concat, total_time_ranges)

    timing_info = Timer.str()
    Timer.reset()

    return TrainResults(cond_cross_entropy, cond_cross_entropy_paper, input_likelihood_concat,
                        rate_tracker=model.output_neuron_cell.rate_tracker,
                        inhibition_tracker=model.inhibition_tracker,
                        learning_rates_tracker=model.stdp_module.learning_rates_tracker,
                        weights_tracker=model.weight_tracker,
                        trained_params=(model.linear.weight.data.clone(), model.linear.bias.data.clone()),
                        neuron_pattern_mapping=neuron_pattern_mapping,
                        timing_info=timing_info)


def test_model(config: TestConfig, data_loader):
    model_config = config.model_config
    stdp_config = model_config.stdp_config

    # Model setup
    encoder, model = init_model(model_config)
    model.linear.weight.data = config.trained_params[0]
    model.linear.bias.data = config.trained_params[1]

    # Used for input likelihood calculation
    input_groups = np.repeat(np.arange(model_config.input_neuron_count // 2), 2)

    # Metric tracking
    total_acc = 0
    total_miss = 0

    pred_option_count = config.distinct_target_count + 1
    confusion_matrix = np.zeros((pred_option_count, pred_option_count))

    total_output_spikes = []
    total_input_spikes = []
    total_time_ranges = [[] for _ in range(config.distinct_target_count)]
    total_time_ranges_ungrouped = []
    input_likelihood_hist = []
    total_targets = []

    model.output_neuron_cell.rate_tracker.is_active = True

    # Test loop
    offset = 0
    osc_phase = None
    state = None
    Timer.reset()
    for i, (data, targets) in enumerate(iter(data_loader)):
        with (Timer('test loop')):
            input_spikes, osc_phase = encoder(data, osc_phase)
            total_input_spikes.append(input_spikes.cpu().numpy())

            targets_np = targets.cpu().numpy()
            time_ranges = encoder.get_time_ranges_for_patterns(targets_np,
                                                               distinct_pattern_count=config.distinct_target_count,
                                                               offset=offset)
            time_ranges_ungrouped = encoder.get_time_ranges(targets_np.shape[0])
            time_ranges_ungrouped_w_offset = encoder.get_time_ranges(targets_np.shape[0], offset=offset)

            for idx, time_range in enumerate(time_ranges):
                total_time_ranges[idx].extend(time_range)
            total_time_ranges_ungrouped.extend(time_ranges_ungrouped_w_offset)

            output_spikes, state, input_psp = model(input_spikes, state=state, train=False)

            with Timer('metric_processing'):
                output_spikes_np = output_spikes.cpu().numpy()
                total_output_spikes.append(output_spikes_np)

                time_offset = encoder.get_time_for_offset(offset)

                pred = get_predictions(output_spikes_np, time_ranges_ungrouped, config.neuron_pattern_mapping)

                confusion_matrix += np.bincount(targets_np * pred_option_count + (pred + 1),
                                                minlength=(pred_option_count ** 2)
                                                ).reshape((pred_option_count, pred_option_count))

                total_acc += np.mean(pred == targets_np)
                total_miss += np.mean(pred == -1)
                total_targets.append(targets_np)

                with Timer('cumulative_counts'):
                    cumulative_counts = get_cumulative_counts_over_time(np.array(output_spikes_np),
                                                                        time_ranges,
                                                                        base_counts=None if i == 0 else
                                                                        cumulative_counts[-1],
                                                                        time_offset=time_offset)

                with Timer('input_likelihood'):
                    weight_np = model.linear.weight.data.cpu().numpy()
                    bias_np = model.linear.bias.data.cpu().numpy()
                    input_psp_np = input_psp.cpu().numpy()

                    input_likelihood = get_input_likelihood(weight_np, bias_np,
                                                            input_psp_np, input_groups, stdp_config.c)
                    input_likelihood_hist.append(input_likelihood)

            offset += data.shape[0]

    total_acc /= len(data_loader)
    total_miss /= len(data_loader)

    normalized_rates, log_total_rates = model.output_neuron_cell.rate_tracker.compute()
    output_rates_np = normalized_rates.cpu().numpy()
    targets_concat = np.concatenate(total_targets, axis=0)
    pred_rates = get_predictions_from_rates(output_rates_np, total_time_ranges_ungrouped,
                                            config.neuron_pattern_mapping)

    total_acc_rate = np.mean(pred_rates == targets_concat)

    joint_probs = get_joint_probabilities_from_counts(cumulative_counts[-1], epsilon=1e-5)
    cond_cross_entropy = normalized_conditional_cross_entropy(joint_probs)
    cond_cross_entropy_paper = normalized_conditional_cross_entropy_paper(joint_probs)

    input_likelihood_concat = np.concatenate(input_likelihood_hist, axis=0)
    average_input_likelihood = np.mean(input_likelihood_concat)

    timing_info = Timer.str()
    Timer.reset()

    if config.print_results:
        print(f"Test Accuracy: {total_acc:.4f}")
        print(f"Test Rate Accuracy: {total_acc_rate:.4f}")
        print(f"Test Miss Rate: {total_miss:.4f}")
        print(f"Test Cross Entropy: {cond_cross_entropy:.4f}")
        print(f"Test Paper Cross Entropy: {cond_cross_entropy_paper:.4f}")
        print(f"Test Average Input Likelihood: {average_input_likelihood:.4f}")

    return TestResults(total_acc, total_acc_rate, total_miss, confusion_matrix,
                       cond_cross_entropy,
                       cond_cross_entropy_paper,
                       rate_tracker=model.output_neuron_cell.rate_tracker,
                       inhibition_tracker=model.inhibition_tracker,
                       average_input_likelihood=average_input_likelihood,
                       total_input_spikes=np.concatenate(total_input_spikes, axis=0),
                       total_output_spikes=np.concatenate(total_output_spikes, axis=0),
                       total_time_ranges=total_time_ranges,
                       timing_info=timing_info)
