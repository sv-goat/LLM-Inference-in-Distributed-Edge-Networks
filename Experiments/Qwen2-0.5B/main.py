"""
Qwen2-0.5B Quantization Experiment

This script implements and evaluates different importance calculation methods for token activations
in the Qwen2-0.5B model. It supports various quantization strategies and analyzes their impact
on model performance using perplexity metrics.
"""

# Requisite imports
from transformers import AutoTokenizer
import json
import pickle
from tqdm import tqdm
from datasets import load_dataset
import torch
import math
from qwen_layer_wise import QwenPointFiveBModel
from transformers import AutoModelForCausalLM
import channel_wise

def get_importance_order(method, attention_map, num_layers, head_weights):
    """
    Calculate importance order of tokens based on attention patterns.

    Args:
        method (str): Importance calculation method ('regular_importance', 'weighted_importance', 'last_row', 'aggregate_till')
        attention_map (list[torch.Tensor]): List of attention maps for each layer
        num_layers (int): Number of layers in the model
        head_weights (list[torch.Tensor]): Weights for each attention head (for weighted_importance)

    Returns:
        list[torch.Tensor]: List of importance scores for each layer

    Importance ordering methods:
    1. Regular:        Here, we take the columnwise mean of the attention map for each head and then average across
                       heads. This can be considered to be the average attention received by each token in the sequence.
    2. Weighted:       This is essentially the same as the first method. However, instead of taking a simple mean across
                       all the heads, we take the weighted mean.
    3. Last Row:       In this method, the importance ordering is the average last row of the attention map across heads.
    4. Aggregate till: In this method, the importance ordering is the running mean of the regular importance
                       till the current layer.
    """
    res = []
    aggregate_importance = 0
    for layer in range(num_layers):
        if method == "regular_importance":
            # We take the average across all heads first.
            avg_across_heads = torch.mean(attention_map[layer], dim = 1)
            # Shape is batch x seq x seq
            # Now, we take the column wise mean
            column_wise_mean = torch.mean(avg_across_heads, dim = 1)
            # Shape is batch x seq
            # Batch dimension will always be 1, so we remove that and append the importance
            # ordering to the result
            column_wise_mean = column_wise_mean.squeeze(0)
            res.append(column_wise_mean)
        elif method == "weighted_importance":
            # Get the attention map for the current layer and the weights for each head
            layer_attention_map = attention_map[layer] # Shape: batch x heads x seq x seq
            layer_head_weights = head_weights[layer] # List of 'heads' tensors, each shape: seq x seq

            # Initialize a tensor to store the weighted sum of attention maps across heads
            # Shape: batch x seq x seq
            weighted_sum_tensor = torch.zeros_like(layer_attention_map[:, 0, :, :])

            # Iterate through each head and apply the corresponding weight
            num_heads = layer_attention_map.shape[1]
            for h in range(num_heads):
                head_attention_map = layer_attention_map[:, h, :, :] # Shape: batch x seq x seq
                head_weight = layer_head_weights[h] # Shape: seq x seq
                # Multiply the head attention map by the head weight (broadcasting over batch)
                weighted_sum_tensor += head_attention_map * head_weight

            # Calculate the column-wise mean of the weighted sum tensor
            column_wise_mean = torch.mean(weighted_sum_tensor, dim=1)

            # Squeeze the batch dimension and append to the result
            res.append(column_wise_mean.squeeze(0))

        elif method == "last_row":
            # Take the last row of the attention map for each head, then average across heads
            last_row_attention = attention_map[layer][:, :, -1, :]
            avg_across_heads = torch.mean(last_row_attention, dim=1)
            # Shape is batch x seq
            avg_across_heads = avg_across_heads.squeeze(0)
            res.append(avg_across_heads)
        elif method == "aggregate_till":
            current_layer_importance = torch.mean(attention_map[layer], dim = 1)
            current_layer_importance = current_layer_importance.squeeze(0)
            current_layer_importance = torch.mean(current_layer_importance, dim = 0)
            aggregate_importance = aggregate_importance + current_layer_importance
            res.append(aggregate_importance / (layer + 1))
        else:
            raise ValueError(f"Unknown method: {method}")

    # Return a list of tensors, where each tensor corresponds to the importance
    # order for a specific layer, according to the chosen method.
    return res

if __name__ == "__main__":
    """
    Main execution block for Qwen2-0.5B quantization experiments.
    
    Loads the model, processes the wikitext dataset, and evaluates different
    quantization strategies based on the specified importance calculation methods.
    """
    # Load the qwen exp params file
    with open("params.json", "r") as f:
        qwen_exp_params = json.load(f)

    # Choose 5 random layers to be considered as partition from the possible layers in qwen 2 0.5 B model
    layers_of_interest = qwen_exp_params["layers_of_interest"]
    ratios = qwen_exp_params["ratios"]

    # The 5 methods
    methods = qwen_exp_params["methods"]

    if methods[0].contains("channel"):
        channel_wise(layers_of_interest, methods)

    else:
        wikitext = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B')
        encodings = tokenizer("\n\n".join(wikitext["text"]), return_tensors="pt")

        device = "cuda"
        qwen_model = QwenPointFiveBModel(device)

            with open('../../attention_head_weights.pkl', 'rb') as f:
            attention_head_weights = pickle.load(f)

        enthu_qwen = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-0.5B', attn_implementation="eager")
        enthu_qwen.eval()
        enthu_qwen.to(device)

        # Get the attention maps at the begining for a given sequence
        max_length = qwen_exp_params["max_length"]
        stride = qwen_exp_params["stride"]
        seq_len = encodings.input_ids.size(1)
        prev_end_loc = 0

        total_num_layers = qwen_model.num_layers

        # Initialize total_nll to accumulate NLL across chunks
        total_nll = [[[0 for _ in range(len(ratios))] for __ in range(len(layers_of_interest))] for ___ in
                     range(len(methods))]
        n_tokens = 0  # Initialize n_tokens

        iterations = 0

        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                base_output = enthu_qwen(input_ids=input_ids, output_attentions=True)
                attention_map = base_output.attentions
            # Calculate importance values once per method per chunk
            importance_values_dict = {}
            for method in methods:
                importance_values_dict[method] = get_importance_order(method, attention_map, total_num_layers,
                                                                      attention_head_weights)
            num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
            batch_size = target_ids.size(0)
            num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift

            for m, method in enumerate(methods):

                importance_values = importance_values_dict[method]
                for l, layer_of_interest in enumerate(layers_of_interest):
                    for r, ratio in enumerate(ratios):
                        neg_log_likelihood = qwen_model.activation_quantization(input_ids, target_ids,
                                                                                importance_values[layer_of_interest],
                                                                                ratio, layer_of_interest)
                        total_nll[m][l][r] += (neg_log_likelihood.item() * num_loss_tokens)

            n_tokens += num_loss_tokens

            prev_end_loc = end_loc

            if iterations % 1000 == 0:
                print(f"Processed {iterations} chunks")
                print(f"Total NLL: {total_nll}")
                print(f"Total tokens: {n_tokens}")
                # Save the total_nll and n_tokens to a file
                with open('total_nll.pkl', 'wb') as f:
                    pickle.dump(total_nll, f)
                with open('n_tokens.pkl', 'wb') as f:
                    pickle.dump(n_tokens, f)

            iterations += 1

            if end_loc == seq_len:
                break
        avg_ppl_results = [[[0. for _ in range(len(ratios))] for __ in range(len(layers_of_interest))] for ___ in
                           range(len(methods))]

        for m, method in enumerate(methods):
            for l, layer_of_interest in enumerate(layers_of_interest):
                for r, ratio in enumerate(ratios):
                    avg_ppl_results[m][l][r] = math.exp(total_nll[m][l][r] / n_tokens)

        with open('avg_ppl_results.pkl', 'wb') as f:
            pickle.dump(avg_ppl_results, f)
