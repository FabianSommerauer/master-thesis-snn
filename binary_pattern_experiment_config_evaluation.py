import json
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from binary_pattern_dataset import BinaryPatternDataset
from my_spike_modules import InhibitionArgs, NoiseArgs, LogFiringRateCalculationMode
from my_utils import set_seed
from train_test_loop import ModelConfig, EncoderConfig, STDPConfig, OutputCellConfig, TrainConfig, TestConfig, \
    evaluate_config

# Set seed
seed = 5464
set_seed(seed)

# Data config
batch_size = 1
num_patterns = 10
num_repeats_train = 100
num_repeats_test = 10
pattern_length = 300
pattern_sparsity = 0.5


def init_binary_pattern_dataset(seed=None):
    # Load data
    binary_train = BinaryPatternDataset(num_patterns, num_repeats_train, pattern_length, pattern_sparsity, seed=seed)
    binary_test = BinaryPatternDataset(num_patterns, num_repeats_test, pattern_length, pattern_sparsity, seed=seed)

    train_loader = DataLoader(binary_train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(binary_test, batch_size=batch_size, shuffle=False)

    # distinct_targets = binary_train.pattern_ids.unique().cpu().numpy()

    return train_loader, test_loader


# Model config
binary_input_variable_cnt = pattern_length
input_neuron_count = binary_input_variable_cnt * 2
output_neuron_count = num_patterns

input_osc_args = None  # BackgroundOscillationArgs(1, 20, -torch.pi / 2)
output_osc_args = None  # BackgroundOscillationArgs(50, 20, -torch.pi / 2)

inhibition_args = InhibitionArgs(2000, 100, 5e-3)  # 1000, 0, 5e-3 (classic); 2000, 100, 5e-3 (adaptive)
noise_args = NoiseArgs(0, 5e-3, 50)

model_config = ModelConfig(
    dt=1e-3,
    input_neuron_count=input_neuron_count,
    output_neuron_count=output_neuron_count,
    sigma=5e-3,

    encoder_config=EncoderConfig(
        presentation_duration=4e-2,
        delay=1e-2,
        active_rate=30,
        inactive_rate=10,
        background_oscillation_args=input_osc_args
    ),
    stdp_config=STDPConfig(
        base_mu=2e-1,  # 5e-1 (classic); 2e-1 (adaptive)
        base_mu_bias=1e-1,  # 5e-1 (classic); 1e-1 (adaptive)
        c=1.,
        time_batch_size=5,
        adaptive=True,
    ),
    output_cell_config=OutputCellConfig(
        inhibition_args=inhibition_args,
        noise_args=noise_args,
        log_firing_rate_calc_mode=LogFiringRateCalculationMode.ExpectedInputCorrected,
        background_oscillation_args=output_osc_args,
    ),

    weight_init=0,  # 0 (classic); 2 (adaptive)
    bias_init=-2  # -2 (classic); 2 (adaptive)
)

train_config = TrainConfig(
    num_epochs=1,
    distinct_target_count=num_patterns,
    print_interval=None,

    model_config=model_config,
)

test_config = TestConfig(
    distinct_target_count=num_patterns,
    model_config=model_config,
    print_results=False,
)

repeats = 5
seeds = [random.randint(0, 10000) for _ in range(repeats)]


def print_eval_results(experiment_name, value, res):
    print("---------------------------------")
    print(f"Experiment: {experiment_name}; Value: {value}")
    print("---------------------------------")
    print(f"Average accuracy: {np.mean(res['accuracy'])}")
    print(f"Std accuracy: {np.std(res['accuracy'])}")
    print(f"Average rate accuracy: {np.mean(res['rate_accuracy'])}")
    print(f"Std rate accuracy: {np.std(res['rate_accuracy'])}")
    print(f"Average miss rate: {np.mean(res['miss_rate'])}")
    print(f"Std miss rate: {np.std(res['miss_rate'])}")
    print(f"Average loss: {np.mean(res['loss'])}")
    print(f"Std loss: {np.std(res['loss'])}")
    print(f"Average loss paper: {np.mean(res['loss_paper'])}")
    print(f"Std loss paper: {np.std(res['loss_paper'])}")
    print(f"Average input log likelihood: {np.mean(res['input_log_likelihood'])}")
    print(f"Std input log likelihood: {np.std(res['input_log_likelihood'])}")
    print("---------------------------------\n")


def set_inhibition_rest(model_config: ModelConfig, value: float):
    model_config.output_cell_config.inhibition_args.inhibition_rest = value


experiment_name = 'inhibition_rest'
param_name = 'Inhibition rest'
set_param_func = set_inhibition_rest
param_values = [0, 25, 50, 75, 100, 125, 150, 175, 200]

# save base config
with open(f'./results/config_eval/{experiment_name}/{experiment_name}_base_config.txt', 'w') as f:
    f.write(str(model_config))

# Train loss
avg_train_loss = None
avg_train_loss_paper = None
train_losses = []
train_losses_paper = []

# Dataframe list
dfs = []

# Run experiment
for val in param_values:
    set_param_func(model_config, val)
    train_config.model_config = model_config
    test_config.model_config = model_config

    eval_results = evaluate_config(train_config, test_config, init_binary_pattern_dataset, seeds=seeds)
    print_eval_results(experiment_name, val, eval_results)

    # update train loss
    train_losses.append(eval_results['avg_train_loss'])
    train_losses_paper.append(eval_results['avg_train_loss_paper'])
    if avg_train_loss is None:
        avg_train_loss = eval_results['avg_train_loss']
    else:
        avg_train_loss += eval_results['avg_train_loss']

    if avg_train_loss_paper is None:
        avg_train_loss_paper = eval_results['avg_train_loss_paper']
    else:
        avg_train_loss_paper += eval_results['avg_train_loss_paper']

    # # save results
    # eval_results['confusion_matrix'] = [mat.tolist() for mat in eval_results['confusion_matrix']]
    # with open(f'./results/config_eval/{experiment_name}/{experiment_name}_{val}_{seed}.json', 'w') as f:
    #     json.dump(eval_results, f, indent=4)

    # add results to dataframe
    new_df = pd.DataFrame({
        'experiment': [experiment_name] * repeats,
        'seed': [seed] * repeats,
        'value': [val] * repeats,
        'accuracy': eval_results['accuracy'],
        'rate_accuracy': eval_results['rate_accuracy'],
        'miss_rate': eval_results['miss_rate'],
        'loss': eval_results['loss'],
        'loss_paper': eval_results['loss_paper'],
        'input_log_likelihood': eval_results['input_log_likelihood'],
        'confusion_matrix_str': [str(mat) for mat in eval_results['confusion_matrix']]
    })
    dfs.append(new_df)

# average train loss
avg_train_loss /= len(param_values)
avg_train_loss_paper /= len(param_values)

# save average train loss
with open(f'./results/config_eval/{experiment_name}/{experiment_name}_{seed}_avg_train_loss.json', 'w') as f:
    json.dump(avg_train_loss.tolist(), f, indent=4)
with open(f'./results/config_eval/{experiment_name}/{experiment_name}_{seed}_avg_train_loss_paper.json', 'w') as f:
    json.dump(avg_train_loss_paper.tolist(), f, indent=4)

df = pd.concat(dfs)
# Save dataframe
df.to_csv(f'./results/config_eval/{experiment_name}/{experiment_name}_{seed}.csv', index=False)

# Plot results as boxplots
df.boxplot(column=['accuracy', 'rate_accuracy', 'miss_rate'],
           by='value', figsize=(20, 10))
plt.ylim([0, 1])
plt.suptitle(param_name)
plt.tight_layout()
plt.savefig(f'./results/config_eval/{experiment_name}/{experiment_name}_{seed}_accuracy.png')
plt.show()

df.boxplot(column=['loss', 'loss_paper'],
           by='value', figsize=(20, 10))
plt.ylim([0, 1])
plt.suptitle(param_name)
plt.tight_layout()
plt.savefig(f'./results/config_eval/{experiment_name}/{experiment_name}_{seed}_loss.png')
plt.show()

df.boxplot(column=['input_log_likelihood'],
           by='value', figsize=(20, 10))
plt.suptitle(param_name)
plt.tight_layout()
plt.savefig(f'./results/config_eval/{experiment_name}/{experiment_name}_{seed}_input_log_likelihood.png')
plt.show()

# plot average train loss
plt.plot(avg_train_loss, label='Train loss')
plt.plot(avg_train_loss_paper, label='Train loss paper')
plt.title('Training')
plt.xlabel('Time')
plt.ylim([0, 1])
plt.tight_layout()
plt.legend()
plt.savefig(f'./results/config_eval/{experiment_name}/{experiment_name}_{seed}_avg_train_loss.png')
plt.show()

# plot each loss
for i in range(len(train_losses)):
    plt.plot(train_losses[i], label=f'{param_name} = {param_values[i]}')
plt.title('Training')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.ylim([0, 1])
plt.tight_layout()
plt.legend()
plt.savefig(f'./results/config_eval/{experiment_name}/{experiment_name}_{seed}_train_loss.png')
plt.show()

for i in range(len(train_losses_paper)):
    plt.plot(train_losses_paper[i], label=f'{param_name} = {param_values[i]}')
plt.title('Training')
plt.xlabel('Time')
plt.ylabel('Paper Loss')
plt.ylim([0, 1])
plt.tight_layout()
plt.legend()
plt.savefig(f'./results/config_eval/{experiment_name}/{experiment_name}_{seed}_train_loss_paper.png')
plt.show()
