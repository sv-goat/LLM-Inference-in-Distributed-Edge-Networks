"""
Quantization Impact Analysis on Pythia 70M Model

This script explores the effects of various quantization strategies on the Pythia 70M language model
by analyzing different importance calculation schemes for token activations. It implements multiple
quantization approaches and evaluates their impact on model performance.

Parameters:
    ratios (list[float]): List of quantization ratios (0 to 10) to apply to activations. The actual value would be 0.1 * ratio.
    layers_of_interest (list[Union[int, str]]): Layers to analyze, can include specific layer
        indices or special values like 'aggregate upto 2', 'maximum aggregation', 'upto ratio'

In aggregate upto 2, we aggregate the importance scores of the  first two layers and then use this combined value.
In upto ratio, we try to only retain the tokens that correspond to 1 - ratio amount of the total importance score.
The rest are quantized.
"""

# Requisite imports
import json
from last_row_exp import last_row_exp
from initial_exp import initial_exp

if __name__ == "__main__":
    with open("params.json", "r") as f:
        params = json.load(f)

    if params["experiment"] == "last_row":
        last_row_exp(params)
    elif params["experiment"] == "initial":
        initial_exp(params)
    else:
        raise ValueError("Invalid experiment type")