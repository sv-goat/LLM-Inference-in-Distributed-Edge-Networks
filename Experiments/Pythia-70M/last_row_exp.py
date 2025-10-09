from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
from tqdm import tqdm
from datasets import load_dataset
import torch
import math
from pythia_model import Pythia70Model

def get_importance_order(method, attention_map, num_layers):
  res = []
  aggregate_importance = 0
  for l in range(num_layers):
    if method == "regular_importance":
      # We take the average across all heads first.
      avg_across_heads = torch.mean(attention_map[l], dim = 1)
      # Shape rn is batch x seq x seq
      # Now, we take the column wise mean
      column_wise_mean = torch.mean(avg_across_heads, dim = 1)
      # Shape rn is batch x seq
      # Batch dimension will always be 1, so we remove that and the importance
      # ordering to the result
      column_wise_mean = column_wise_mean.squeeze(0)
      res.append(column_wise_mean)
    elif method == "last_row":
      # Take the last row of the attention map for each head, then average across heads
      last_row_attention = attention_map[l][:, :, -1, :]
      avg_across_heads = torch.mean(last_row_attention, dim=1)
      # Shape is batch x seq
      avg_across_heads = avg_across_heads.squeeze(0)
      res.append(avg_across_heads)
    elif method == "aggregate_till":
      # This is basically going to be the running mean of importance till this
      # layer
      # first, we calculate current layer importance
      current_layer_importance = torch.mean(attention_map[l], dim = 1)
      current_layer_importance = current_layer_importance.squeeze(0)
      current_layer_importance = torch.mean(current_layer_importance, dim = 0)
      aggregate_importance = aggregate_importance + current_layer_importance
      res.append(aggregate_importance / (l + 1))
    elif method == "random":
      # Random importance values - generate random values that will result in random token selection
      # when sorted in ascending order (least important first)
      seq_len = attention_map[l].shape[-1]
      random_importance = torch.rand(seq_len)
      res.append(random_importance)
    else:
      raise ValueError(f"Unknown method: {method}")

  # Return a list of tensors, where each tensor corresponds to the importance
  # order for a specific layer, according to the chosen method.
  return res

def last_row_exp(params):

    wikitext = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

    # Load the pythia 70M model tokenizer
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')

    # Join the entire wikitext test corpus
    encodings = tokenizer("\n\n".join(wikitext["text"]), return_tensors="pt")

    device = "cuda"

    ratios = params["ratios"]

    layers_of_interest = params["layers_of_interest"]

    # The 5 methods
    methods = params["methods"]

    pythia_model = Pythia70Model(device, ratios)

    enthu_pythia = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-70m', attn_implementation="eager")
    enthu_pythia.eval()
    enthu_pythia.to(device)

    max_length = pythia_model.model.config.max_position_embeddings
    stride = 32
    seq_len = encodings.input_ids.size(1)
    prev_end_loc = 0

    total_num_layers = pythia_model.num_layers

    total_nll = [[[0 for _ in range(len(ratios))] for __ in range(len(layers_of_interest))] for ___ in
                 range(len(methods))]
    n_tokens = 0

    iterations = 0

    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            # Issue is that I need eager attn here and not down there.
            base_output = enthu_pythia(input_ids=input_ids, output_attentions=True)
            attention_map = base_output.attentions
        # Calculate importance values once per method per chunk
        importance_values_dict = {}
        for method in methods:
            importance_values_dict[method] = get_importance_order(method, attention_map, total_num_layers)
        num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift

        for m, method in enumerate(methods):
            importance_values = importance_values_dict[method]
            for l, layer_of_interest in enumerate(layers_of_interest):
                for r, ratio in enumerate(ratios):
                    neg_log_likelihood = pythia_model.activation_quantization(input_ids, target_ids,
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
            with open('total_nll_pythia_70m.pkl', 'wb') as f:
                pickle.dump(total_nll, f)
            with open('n_tokens_pythia_70m.pkl', 'wb') as f:
                pickle.dump(n_tokens, f)

        iterations += 1

        if end_loc == seq_len:
            break

    avg_ppl_results = [[[0 for _ in range(len(ratios))] for __ in range(len(layers_of_interest))] for ___ in
                       range(len(methods))]

    for m, method in enumerate(methods):
        for l, layer_of_interest in enumerate(layers_of_interest):
            for r, ratio in enumerate(ratios):
                avg_ppl_results[m][l][r] = math.exp(total_nll[m][l][r] / n_tokens)

    print("Average ppl results", avg_ppl_results)

    with open('avg_ppl_results_pythia_70m.pkl', 'wb') as f:
        pickle.dump(avg_ppl_results, f)







