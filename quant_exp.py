# Requisite imports
import json
from tqdm import tqdm
from datasets import load_dataset
import torch
from pythia_model import Pythia70Model
import numpy as np
import math


# Identify the dataset
# Let us use wikitext or something?

# We will use the  pile test dataset in a streaming way

def extract_attentions(attention_weights, interesting_layers):
    # Define a dictionary called per_seq_ordering that corresponds to the different layers mentioned in layers of interest.
    calculated_orderings = {}
    for layer in interesting_layers:
      if isinstance(layer, str) and layer == 'aggregate upto 2':
        aggregate_column_wise_means = 0
        for i in range(3):
          aggregate_attn_weights = torch.mean(attention_weights[i], dim=1)
          column_wise_mean = torch.mean(aggregate_attn_weights, dim=1)
          column_wise_mean = column_wise_mean.squeeze(0)
          aggregate_column_wise_means += column_wise_mean
        aggregate_column_wise_means = aggregate_column_wise_means / 3
        calc = torch.argsort(aggregate_column_wise_means, descending=False).cpu().numpy()
        calculated_orderings[layer] = calc
      elif layer == 'maximum aggregation':
        # Create zeroes of shape = model dimension
        aggregate_column_wise_means = torch.zeros(attention_weights[0].shape[-1])
        for i in range(3):
          aggregate_attn_weights = torch.mean(attention_weights[i], dim=1)
          column_wise_mean = torch.mean(aggregate_attn_weights, dim=1)
          column_wise_mean = column_wise_mean.squeeze(0).detach().cpu()
          aggregate_column_wise_means = torch.maximum(aggregate_column_wise_means, column_wise_mean)
        # Get the ordering
        calc = torch.argsort(aggregate_column_wise_means, descending=False).numpy()
        calculated_orderings[layer] = calc
      elif layer == 'upto ratio':
        # ggregate across heads
        # We only consider second layer for upto ratio ( boundary layer )
        aggregate_attn_weights = torch.mean(attention_weights[2], dim=1)
        # Mean the aggregates column wise
        # Make sure this is doing what youa re expecting it to do.
        column_wise_mean = torch.mean(aggregate_attn_weights, dim=1)
        # Remove batch dimensin
        column_wise_mean = column_wise_mean.squeeze(0)
        calculated_orderings[layer] = column_wise_mean.detach().cpu().numpy()
      else:
        # ggregate across heads
        aggregate_attn_weights = torch.mean(attention_weights[layer], dim=1)
        # Mean the aggregates column wise
        # Make sure this is doing what youa re expecting it to do.
        column_wise_mean = torch.mean(aggregate_attn_weights, dim=1)
        # Remove batch dimensin
        column_wise_mean = column_wise_mean.squeeze(0)
        calc = torch.argsort(column_wise_mean, descending=False).cpu().numpy()
        calculated_orderings[layer] = calc
    return calculated_orderings

if __name__ == "__main__":
    wikitext = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

    # Define the model and the tokenizer
    device = "cuda"
    ratios = [i for i in range(5)]
    pythia_model = Pythia70Model(device, ratios)

    encodings = pythia_model.tokenizer("\n\n".join(wikitext["text"]), return_tensors="pt")

    max_length = pythia_model.model.config.max_position_embeddings
    stride = 32
    seq_len = encodings.input_ids.size(1)

    layers_of_interest = [2, 3, 5, 'aggregate upto 2', 'maximum aggregation', 'upto ratio']

    total_nlls = {}
    for l in layers_of_interest:
        total_nlls[l] = {}
        for ratio in ratios:
            total_nlls[l][ratio] = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = pythia_model.model(input_ids, output_attentions=True)

        per_seq_ordering = extract_attentions(outputs.attentions, layers_of_interest)
        for l in layers_of_interest:
            # This is ascending order of token orderings. So, initial tokens are the least important
            ordering = per_seq_ordering[l]
            # Create a batch of inputs of size equal to number of ratios
            size = len(ratios)
            batched_encodings = [input_ids.squeeze(0) for _ in range(size)]
            batched_input_tensor = torch.stack(batched_encodings, dim=0)
            if isinstance(l, str) and 'upto ratio' in l:
                batched_nlls = pythia_model.batched_top_rho_quantization(batched_input_tensor, ordering,
                                                                         target=target_ids, quant_layer=2,
                                                                         do_quant=True)
            else:
                batched_nlls = pythia_model.batched_quantization_helper(batched_input_tensor, ordering,
                                                                        target=target_ids, quant_layer=2, do_quant=True)
            for ratio in ratios:
                total_nlls[l][ratio].append(batched_nlls[ratio].item())

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    # Compute ppl
    for l in layers_of_interest:
        for ratio in ratios:
            total_nlls[l][ratio] = np.mean(total_nlls[l][ratio])
            ppl = math.exp(total_nlls[l][ratio])
            print("Layer", l, "Ratio", 0.1 * ratio, "PPL", ppl)

    with open('corrected_total_nlls_new_ppl.json', 'w') as fp:
        json.dump(total_nlls, fp)
