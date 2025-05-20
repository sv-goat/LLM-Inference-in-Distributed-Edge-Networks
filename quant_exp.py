# Requisite imports
import json
from tqdm import tqdm
from datasets import load_dataset
import torch

from pythia_model import Pythia70Model


# Identify the dataset
# Let us use wikitext or something?

# We will use the  pile test dataset in a streaming way

def extract_attentions(attention_weights, number_of_layers):
    per_seq_ordering = [[] for _ in range(number_of_layers)]
    for layer in range(number_of_layers):
        # Aggregate across heads
        aggregate_attn_weights = torch.mean(attention_weights[layer], dim=1)
        # Mean the aggregates column wise
        column_wise_mean = torch.mean(aggregate_attn_weights, dim=1)
        # Remove batch dimension
        column_wise_mean = column_wise_mean.squeeze(0)
        # Get the ordering
        calculated_ordering = torch.argsort(column_wise_mean, descending=False).cpu().numpy()
        per_seq_ordering[layer] = calculated_ordering
    return per_seq_ordering

if __name__ == "__main__":
    wikitext = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    # Define the model and the tokenizer
    model = Pythia70Model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # For seq in wiktext:
    num_layers = 6
    layers_of_interest = [2, 3, 5]
    ratios = [r for r in range(10)]
    total_nlls = {}
    for l in layers_of_interest:
        total_nlls[l] = {}
        for ratio in ratios:
            total_nlls[l][ratio] = 0
    num_tokens = 0
    num_seq = 0
    for seq in tqdm(wikitext):
        num_seq += 1
        do_quant = True
        quant_layer = 2
        # length should be number of tokens actually
        seq = seq["text"]
        tokens = model.tokenizer(seq, return_tensors="pt").input_ids
        if tokens.shape[1] < 1:
            continue
        length = tokens.shape[1]
        tokens = tokens.to(device)
        for end_mark in range(1, length):
            num_tokens += 1
            sub_seq = tokens[:, :end_mark]
            outputs = model.model(input_ids=sub_seq, output_attentions=True)

            # Get the base ppl

            # extract attentions
            per_seq_ordering = extract_attentions(outputs.attentions, num_layers)

            for l in layers_of_interest:
                # This is ascending order of token orderings. So, initial tokens are the least important
                ordering = per_seq_ordering[l]
                # Create a batch of inputs of size 10
                batched_inputs = [sub_seq.squeeze(0) for _ in range(10)]
                batched_input_tensor = torch.stack(batched_inputs, dim=0)
                batched_nlls = model.batched_quantization_helper(batched_input_tensor, ordering,
                                                                 target=tokens[:, end_mark], quant_layer=quant_layer,
                                                                 do_quant=do_quant)
                for ratio in ratios:
                    total_nlls[l][ratio] += batched_nlls[ratio].item()
    # Save the nlls, number of tokens and the number of sequences
    with open("nlls.json", "w") as f:
        json.dump(total_nlls, f)
    print("Number of tokens", num_tokens)
    print("Number of sequences", num_seq)
    # Average across all the tokens
    for l in layers_of_interest:
        for ratio in ratios:
            total_nlls[l][ratio] /= num_tokens
            print("Layer", l, "Ratio", 0.1 * ratio, "NLL", total_nlls[l][ratio])
            # Calculate the perplexity
            perplexity = torch.exp(total_nlls[l][ratio])
            print("Layer", l, "Ratio", 0.1 * ratio, "Perplexity", perplexity)
            total_nlls[l][ratio] = perplexity

    # Save and download the perplexity
    with open("ovr_res.json", "w") as f:
        json.dump(total_nlls, f)





