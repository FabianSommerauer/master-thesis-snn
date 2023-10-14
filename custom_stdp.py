from typing import Literal, Tuple
import torch


def apply_bayesian_stdp(
    input_psp: torch.Tensor,
    output_spikes: torch.Tensor,
    weights: torch.Tensor,
    biases: torch.Tensor,
    c: float = 1,
    mu: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """STDP step for bayesian computation.
    Input:
        input_psp (torch.Tensor): Postsynaptic potential induced by each input neuron during single time step; shape: (batch, input_count)
        output_spikes (torch.Tensor): Spikes of the output neurons at single time step; shape: (batch, output_count)
        weights (torch.Tensor): Weight tensor of linear layer connecting pre- and postsynaptic layers; shape: (output_count, input_count)
        biases (torch.Tensor): Bias tensor of linear layer connecting pre- and postsynaptic layers; shape: (output_count)
        c (float): constant determining shift of final weights
        mu (float): learning rate
    Output:
        weights (torch.tensor): Updated weights
        biases (STDPState): Updated biases
    """

    # (output_count)
    out_dims = output_spikes.dim()
    if out_dims == 1:
        total_out_spikes = output_spikes
    else:
        total_out_spikes = torch.sum(output_spikes, tuple(range(out_dims-1)))

    # STDP weight update
    # only applies to active neuron
    dw = torch.einsum('...o,...i->oi', output_spikes, input_psp) * c * torch.exp(-weights) - total_out_spikes[:, None]

    # applies to all neurons
    db = torch.exp(-biases) * total_out_spikes - 1

    weights = weights + mu * dw
    biases = biases + mu * db

    return weights, biases