import torch
from transformers import AutoTokenizer
from transformers.models.qwen2 import modeling_qwen2

from lxt.efficient import monkey_patch
from datasets import load_dataset

from tqdm import tqdm

import random

import json

def hook_attention_output(module, input, output):
    # Assuming the attention output is the second element in the output tuple
    if isinstance(output, tuple) and len(output) > 1:
        # save the attention output and make sure the gradient is also saved
        # This assumes the attention weights are the second element in the tuple
        attention_output = output[1]
        module.attention_output = attention_output
        module.attention_output.retain_grad() if module.attention_output.requires_grad else None

if __name__ == "__main__":
    monkey_patch(modeling_qwen2, verbose=False)
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B')

    wikitext_test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

    encodings = tokenizer("\n\n".join(wikitext_test["text"]), return_tensors="pt", add_special_tokens=True)

    # Load the params
    with open("params.json", "r") as f:
        params = json.load(f)

    max_length = params["max_length"]
    stride = params["stride"]
    seq_len = encodings.input_ids.size(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = modeling_qwen2.Qwen2ForCausalLM.from_pretrained('Qwen/Qwen2-0.5B', device_map='cuda',
                                                            attn_implementation="eager")

    # optional gradient checkpointing to save memory (2x forward pass)
    model.train()
    model.gradient_checkpointing_enable()

    # deactive gradients on parameters to save memory
    for param in model.parameters():
        param.requires_grad = False

    # apply hooks
    for layer in model.model.layers:
        # We need to hook the attention module within each layer
        layer.self_attn.register_forward_hook(hook_attention_output)

    prev_end_loc = 0

    # trace relevance through layers
    attention_head_weights = [[0 for i in range(model.model.layers[0].self_attn.attention_output.shape[1])] for j in
                              range(len(model.model.layers))]

    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        input_embeds = model.get_input_embeddings()(input_ids)
        output_logits = model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False,
                              output_attentions=True).logits

        max_logits, max_indices = torch.max(output_logits[:, -1, :], dim=-1)
        max_logits.backward(max_logits)

        for i, layer in enumerate(model.model.layers):
            # Extract attention weights from the hooked module
            if hasattr(layer.self_attn, 'attention_output') and layer.self_attn.attention_output is not None:
                # Assuming attention_output is (batch_size, num_heads, sequence_length, sequence_length)
                # We'll take the mean across heads and batch size for simplicity
                # Get the relevance scores corresponding to the attention neurons
                relevance = (layer.self_attn.attention_output * layer.self_attn.attention_output.grad).float()
                # Add the values to the list
                for batch in relevance:
                    for j, head in enumerate(batch):
                        # Get the sum of all the sequence_length x sequence_length relevance values
                        tensor = head.float().sum()
                        # Write the scalar value to heads
                        attention_head_weights[i][j] += tensor.cpu().item()


            else:
                print("No attention output found for layer", i)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    # Normalize the attention head weights for each layer
    for layer_weights in attention_head_weights:
        # Calculate the sum of weights for the current layer
        layer_sum = sum(layer_weights)

        # Normalize the heads within the current layer
        for i in range(len(layer_weights)):
            # Avoid division by zero in case all weights in a layer are zero
            layer_weights[i] /= layer_sum if layer_sum != 0 else 1e-9

    # Now attention_head_weights contains the weights normalized within each layer, summing to 1
    print("Normalized relevance sccores for attention head (per layer, summing to 1):")
    # Print everything.
    for i in range(len(attention_head_weights)):
        print(f"Layer {i}: {attention_head_weights[i]}")

    # Save the weights
    with open("attention_head_weights.json", "w") as f:
        json.dump(attention_head_weights, f)


