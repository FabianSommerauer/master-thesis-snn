from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch

import custom_stdp
from my_spike_modules import BinaryTimedPSP, EfficientBayesianSTDPModel, LogFiringRateCalculationMode, NoiseArgs, \
    InhibitionArgs, EfficientStochasticOutputNeuronCell, SpikePopulationGroupBatchToTimeEncoder, \
    BackgroundOscillationArgs, InputBackgroundOscillationArgs
from my_timing_utils import Timer
from my_trackers import SpikeRateTracker, InhibitionStateTracker, LearningRatesTracker, WeightsTracker
from my_utils import normalized_conditional_cross_entropy_paper, normalized_conditional_cross_entropy, \
    get_joint_probabilities_from_counts, get_cumulative_counts_over_time, get_predictions, \
    get_predictions_from_rates, get_input_log_likelihood, \
    get_neuron_pattern_mapping_from_cumulative_counts, set_seed


@dataclass
class EncoderConfig:
    presentation_duration: float
    delay: float
    active_rate: float
    inactive_rate: float
    background_oscillation_args: InputBackgroundOscillationArgs | None = None


@dataclass
class STDPMethodConfig:
    pass


@dataclass
class STDPClassicConfig(STDPMethodConfig):
    type: str = 'classic'
    base_mu: float = 1.
    base_mu_bias: float = 1.


@dataclass
class STDPAdaptiveConfig(STDPMethodConfig):
    type: str = 'adaptive'
    base_mu: float = 5e-1
    base_mu_bias: float = 5e-1
    min_mu: float = 1e-6
    min_mu_bias: float = 1e-6
    max_delta: float = 1e0


@dataclass
class STDPConfig:
    c: float
    time_batch_size: int
    method: STDPMethodConfig = None


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
    single_metric_per_batch: bool = False


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
    input_log_likelihood_hist: np.array

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
    average_input_log_likelihood: float

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

    method = stdp_config.method
    if isinstance(method, STDPAdaptiveConfig):
        stdp_module = custom_stdp.BayesianSTDPAdaptive(model_config.input_neuron_count,
                                                       model_config.output_neuron_count,
                                                       time_batch_size=stdp_config.time_batch_size,
                                                       base_mu=method.base_mu,
                                                       base_mu_bias=method.base_mu_bias,
                                                       min_mu_weights=method.min_mu,
                                                       min_mu_bias=method.min_mu_bias,
                                                       max_delta=method.max_delta,
                                                       c=stdp_config.c, collect_history=True)
    elif isinstance(method, STDPClassicConfig):
        stdp_module = custom_stdp.BayesianSTDPClassic(model_config.output_neuron_count, c=stdp_config.c,
                                                      base_mu=method.base_mu,
                                                      base_mu_bias=method.base_mu_bias,
                                                      time_batch_size=stdp_config.time_batch_size,
                                                      collect_history=True)
    else:
        raise ValueError(f"Unknown STDP method: {method}")

    output_cell = EfficientStochasticOutputNeuronCell(inhibition_args=output_cell_config.inhibition_args,
                                                      noise_args=output_cell_config.noise_args,
                                                      log_firing_rate_calc_mode=output_cell_config.log_firing_rate_calc_mode,
                                                      background_oscillation_args=output_cell_config.background_oscillation_args,
                                                      dt=dt, collect_rates=True)

    model = EfficientBayesianSTDPModel(model_config.input_neuron_count, model_config.output_neuron_count,
                                       BinaryTimedPSP(sigma, dt),
                                       multi_step_output_neuron_cell=output_cell,
                                       stdp_module=stdp_module, track_states=True)

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
    total_input_values = []

    # Training
    offset = 0
    osc_phase = None
    state = None
    Timer.reset()
    for epoch in range(num_epochs):
        with Timer('training loop'):
            for i, (data, targets) in enumerate(iter(data_loader)):
                input_spikes, osc_phase = encoder(data, osc_phase)

                output_spikes, state = model(input_spikes, state=state, train=True)

                with Timer('metric_processing'):
                    time_ranges = encoder.get_time_ranges_for_patterns(targets,
                                                                       distinct_pattern_count=config.distinct_target_count,
                                                                       offset=offset)

                    output_spikes_np = output_spikes.cpu().numpy()
                    time_offset = encoder.get_time_for_offset(offset)

                    if config.single_metric_per_batch:
                        total_input_values.append(data[None, -1].cpu().numpy())
                    else:
                        total_input_values.append(data.cpu().numpy())

                    with Timer('cumulative_counts'):
                        cumulative_counts = get_cumulative_counts_over_time(np.array(output_spikes_np),
                                                                            time_ranges,
                                                                            base_counts=None if i == 0 else
                                                                            cumulative_counts[-1],
                                                                            time_offset=time_offset)
                        if config.single_metric_per_batch:
                            cumulative_counts_hist.append(cumulative_counts[None, -1])
                        else:
                            cumulative_counts_hist.append(cumulative_counts)

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

    with Timer('cross_entropy'):
        cumulative_counts_concat = np.concatenate(cumulative_counts_hist, axis=0)
        joint_probs = get_joint_probabilities_from_counts(cumulative_counts_concat)
        cond_cross_entropy = normalized_conditional_cross_entropy(joint_probs)
        cond_cross_entropy_paper = normalized_conditional_cross_entropy_paper(joint_probs)

    with Timer('neuron_pattern_mapping'):
        neuron_pattern_mapping = get_neuron_pattern_mapping_from_cumulative_counts(cumulative_counts_concat[-1])

    with Timer('input_log_likelihood'):
        total_input_log_likelihood = []

        weights, biases = model.weight_tracker.compute()
        weights_np = weights.cpu().numpy()
        biases_np = biases.cpu().numpy()
        for i in range(len(total_input_values)):
            input_log_likelihood = get_input_log_likelihood(weights_np[i], biases_np[i],
                                                            total_input_values[i], stdp_config.c)
            total_input_log_likelihood.append(input_log_likelihood)

    timing_info = Timer.str()
    Timer.reset()

    return TrainResults(cond_cross_entropy, cond_cross_entropy_paper,
                        input_log_likelihood_hist=np.concatenate(total_input_log_likelihood, axis=0),
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

    # Metric tracking
    total_acc = 0
    total_miss = 0

    pred_option_count = config.distinct_target_count + 1
    confusion_matrix = np.zeros((pred_option_count, pred_option_count))

    total_output_spikes = []
    total_input_spikes = []
    total_time_ranges = [[] for _ in range(config.distinct_target_count)]
    total_time_ranges_ungrouped = []
    total_input_values = []
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

            output_spikes, state = model(input_spikes, state=state, train=False)

            with Timer('metric_processing'):
                targets_np = targets.cpu().numpy()
                time_ranges = encoder.get_time_ranges_for_patterns(targets_np,
                                                                   distinct_pattern_count=config.distinct_target_count,
                                                                   offset=offset)
                time_ranges_ungrouped = encoder.get_time_ranges(targets_np.shape[0])
                time_ranges_ungrouped_w_offset = encoder.get_time_ranges(targets_np.shape[0], offset=offset)

                for idx, time_range in enumerate(time_ranges):
                    total_time_ranges[idx].extend(time_range)
                total_time_ranges_ungrouped.extend(time_ranges_ungrouped_w_offset)

                total_input_spikes.append(input_spikes.cpu().numpy())
                output_spikes_np = output_spikes.cpu().numpy()
                total_output_spikes.append(output_spikes_np)

                total_input_values.append(data.cpu().numpy())

                time_offset = encoder.get_time_for_offset(offset)

                pred = get_predictions(output_spikes_np, time_ranges_ungrouped, config.neuron_pattern_mapping)

                confusion_matrix += np.bincount(pred_option_count * (targets_np + 1) + (pred + 1),
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

            offset += data.shape[0]

    total_acc /= len(data_loader)
    total_miss /= len(data_loader)

    normalized_rates, log_total_rates = model.output_neuron_cell.rate_tracker.compute()
    output_rates_np = normalized_rates.cpu().numpy()
    targets_concat = np.concatenate(total_targets, axis=0)
    pred_rates = get_predictions_from_rates(output_rates_np, total_time_ranges_ungrouped,
                                            config.neuron_pattern_mapping)

    total_acc_rate = np.mean(pred_rates == targets_concat)

    joint_probs = get_joint_probabilities_from_counts(cumulative_counts[-1], epsilon=1e-1)
    cond_cross_entropy = normalized_conditional_cross_entropy(joint_probs)
    cond_cross_entropy_paper = normalized_conditional_cross_entropy_paper(joint_probs)

    input_concat = np.concatenate(total_input_values, axis=0)
    weights_np = model.linear.weight.data.cpu().numpy()
    biases_np = model.linear.bias.data.cpu().numpy()
    input_log_likelihood = get_input_log_likelihood(weights_np, biases_np, input_concat, stdp_config.c)

    average_input_log_likelihood = np.mean(input_log_likelihood)

    timing_info = Timer.str()
    Timer.reset()

    if config.print_results:
        print(f"Test Accuracy: {total_acc * 100:.4f}%")
        print(f"Test Rate Accuracy: {total_acc_rate * 100:.4f}%")
        print(f"Test Miss Rate: {total_miss * 100:.4f}%")
        print(f"Test Cross Entropy: {cond_cross_entropy:.4f}")
        print(f"Test Paper Cross Entropy: {cond_cross_entropy_paper:.4f}")
        print(f"Test Average Input Log Likelihood: {average_input_log_likelihood:.4f}")

    return TestResults(total_acc, total_acc_rate, total_miss, confusion_matrix,
                       cond_cross_entropy,
                       cond_cross_entropy_paper,
                       rate_tracker=model.output_neuron_cell.rate_tracker,
                       inhibition_tracker=model.inhibition_tracker,
                       average_input_log_likelihood=average_input_log_likelihood,
                       total_input_spikes=np.concatenate(total_input_spikes, axis=0),
                       total_output_spikes=np.concatenate(total_output_spikes, axis=0),
                       total_time_ranges=total_time_ranges,
                       timing_info=timing_info)


def evaluate_config(train_config: TrainConfig, test_config: TestConfig,
                    init_dataset_func, seeds):
    repeats = len(seeds)

    metrics = {
        'accuracy': [],
        'rate_accuracy': [],
        'miss_rate': [],
        'loss': [],
        'loss_paper': [],
        'input_log_likelihood': [],
        'confusion_matrix': [],
        'avg_train_loss': None,
        'avg_train_loss_paper': None
    }

    for i in range(repeats):
        set_seed(seeds[i])

        train_loader, test_loader = init_dataset_func(seeds[i])

        # Train
        train_results = train_model(train_config, train_loader)

        trained_params = train_results.trained_params
        neuron_pattern_mapping = train_results.neuron_pattern_mapping

        # Test
        test_config.trained_params = trained_params
        test_config.neuron_pattern_mapping = neuron_pattern_mapping

        test_results = test_model(test_config, test_loader)

        # Get metrics
        metrics['accuracy'].append(test_results.accuracy)
        metrics['rate_accuracy'].append(test_results.rate_accuracy)
        metrics['miss_rate'].append(test_results.miss_rate)
        metrics['loss'].append(test_results.cross_entropy)
        metrics['loss_paper'].append(test_results.cross_entropy_paper)
        metrics['input_log_likelihood'].append(test_results.average_input_log_likelihood)
        metrics['confusion_matrix'].append(test_results.confusion_matrix)

        if metrics['avg_train_loss'] is None:
            metrics['avg_train_loss'] = train_results.cross_entropy_hist
        else:
            metrics['avg_train_loss'] += train_results.cross_entropy_hist

        if metrics['avg_train_loss_paper'] is None:
            metrics['avg_train_loss_paper'] = train_results.cross_entropy_paper_hist
        else:
            metrics['avg_train_loss_paper'] += train_results.cross_entropy_paper_hist

    metrics['avg_train_loss'] /= repeats
    metrics['avg_train_loss_paper'] /= repeats

    return metrics
