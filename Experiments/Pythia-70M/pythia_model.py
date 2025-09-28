"""
Pythia 70M Model Wrapper for Quantization Experiments

This module provides a wrapper around the Pythia 70M model, specifically designed
for quantization experiments. It implements methods for quantization-aware forward
passes and loss calculation.
"""

# Requisite imports
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Pythia70Model:
    """
    A wrapper class for the Pythia 70M model that supports quantization experiments.

    This class provides an interface to the Pythia 70M model with additional functionality
    for quantization experiments, including batched processing and custom loss calculation.

    Args:
        device (str): The device to run the model on ('cuda' or 'cpu')
        ratios (list[int]): List of quantization ratios to evaluate (0-10, where 1 = 10%)
    """
    def __init__(self, device, ratios) -> None:
        self.model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-70m', attn_implementation = "sdpa")
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
        self.tokenizer = tokenizer
        self.model.to(device)
        self.device = device
        self.num_layers = len(self.model.base_model.layers)
        self.gpt_neox = self.model.base_model
        self.final_layer_norm = self.model.gpt_neox.final_layer_norm
        self.rotary_emb = self.gpt_neox.rotary_emb
        self.ratios = ratios

    def calculate_batched_nll(self, hidden_states, target):
        with torch.no_grad():
            post_norm = self.final_layer_norm(hidden_states)
            logits = self.model.embed_out(post_norm)
            # logits shape: (batch_size, seq_len, vocab_size)
            # targets are just the inputs. so, there is a need to shift them by 1
            target = target[:, 1:]
            # Since we do not have targets for the last token, we need to shift those as well
            logits = logits[:, :-1, :]

            # Get separate NLLs for each batch.
            batched_nll = []
            for ratio in self.ratios:
                cross_entropy = torch.nn.functional.cross_entropy(logits[ratio, :, :].view(-1, logits.size(-1)),
                                                                  target.view(-1), ignore_index=- 100)
                batched_nll.append(cross_entropy)

            # Return the scalar negative log likelihood
            return batched_nll


    @staticmethod
    def simulate_quantization(self, tokens_to_quantize, hidden_states, ratio):
        for token in tokens_to_quantize:
            activation_for_token = hidden_states[ratio, token, :]
            scale = torch.max(activation_for_token) - torch.min(activation_for_token)
            zero_point = torch.min(activation_for_token)
            quantized_activation_for_token = torch.quantize_per_tensor(activation_for_token, scale,
                                                                       zero_point, torch.qint8)
            unquantized_activation_for_token = torch.dequantize(quantized_activation_for_token)

            hidden_states[ratio, token, :] = unquantized_activation_for_token
        return hidden_states

    # Implementation of upto ratio method from params.json
    def batched_top_rho_quantization(self, batched_input_tokens, distribution, target, quant_layer=2, do_quant=True):
        # input ids have shape batch size x num tokens
        batched_input_tokens = batched_input_tokens.to(self.device)
        gpt_neox = self.model.base_model
        rotary_emb = gpt_neox.rotary_emb

        with torch.no_grad():
            hidden_states = gpt_neox.embed_in(batched_input_tokens)
            # Create position embeddings of shape batch_size , seq _len
            # Value of the position embeddings should be between 0 and the number of positions -1
            position_ids = torch.arange(batched_input_tokens.size(1), dtype=torch.long,
                                        device=batched_input_tokens.device).unsqueeze(0).expand(
                batched_input_tokens.size(0), -1)
            position_embeddings = rotary_emb(hidden_states, position_ids)
            # Six layers present in Pythia 70M, hence hardcoding 6 here.
            for i in range(6):
                layer = gpt_neox.layers[i]
                hidden_states = layer(hidden_states, position_embeddings=position_embeddings)[0]
                # Shape at this stage
                # ( batch, num_input_tokens, model_dim = 512 )

                if i == quant_layer and do_quant:
                    for ratio in self.ratios:
                        # First, let us define the threshold that we want to reach.
                        threshold = 1 - 0.1 * ratio
                        # Now, we want to choose the fewest indices that can add upto this threshold from the
                        # given distribution
                        sorted_distribution_with_indices = sorted(enumerate(distribution), key=lambda x: x[1],
                                                                  reverse=True)
                        total = 0
                        final_index = -1
                        for index, value in sorted_distribution_with_indices:
                            if total >= threshold:
                                break
                            total += value
                            final_index = index

                        tokens_to_quantize = sorted_distribution_with_indices[final_index + 1:]
                        tokens_to_quantize = [t[0] for t in tokens_to_quantize]

                        # Now that we have the tokens to quantize, the rest of the steps are the same.
                        hidden_states = self.simulate_quantization(tokens_to_quantize, hidden_states, ratio)

            return self.calculate_batched_nll( hidden_states, target)

    def batched_quantization_helper(self, batched_input_tokens, ordering, target, quant_layer=2, do_quant=True):
        # input ids have shape batch size x num tokens
        batched_input_tokens = batched_input_tokens.to(self.device)

        gpt_neox = self.model.base_model
        rotary_emb = gpt_neox.rotary_emb

        # Autoregressively generate new token
        with torch.no_grad():
            # Get the hidden states
            hidden_states = gpt_neox.embed_in(batched_input_tokens)
            # Create position embeddings of shape batch_size , seq _len
            # Value of the position embeddings should be between 0 and the number of positions -1
            position_ids = torch.arange(batched_input_tokens.size(1), dtype=torch.long,
                                        device=batched_input_tokens.device).unsqueeze(0).expand(
                batched_input_tokens.size(0), -1)
            position_embeddings = rotary_emb(hidden_states, position_ids)
            # 6 = number of layers in Pythia 70M
            for i in range(6):
                layer = gpt_neox.layers[i]
                hidden_states = layer(hidden_states, position_embeddings=position_embeddings)[0]
                # Shape right now
                # ( batch, num_input_tokens, model_dim = 512 )

                if i == quant_layer and do_quant:
                    for ratio in self.ratios:
                        tokens_to_quantize = ordering[:int(0.1 * ratio * len(ordering))]
                        hidden_states = self.simulate_quantization(tokens_to_quantize, hidden_states, ratio)

            return self.calculate_batched_nll( hidden_states, target)

    def move_to_device(self):
        self.model.to(self.device)

    def remove_from_device(self):
        self.model.to("cpu")

    def activation_quantization(self, batched_input_tokens, target, importance_values, ratio, layer_of_interest):
        batched_input_tokens = batched_input_tokens.to(self.device)

        with torch.no_grad():
            hidden_states = self.gpt_neox.embed_in(batched_input_tokens)
            position_ids = torch.arange(batched_input_tokens.size(1), dtype=torch.long,
                                        device=batched_input_tokens.device).unsqueeze(0).expand(
                batched_input_tokens.size(0), -1)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            # How many layers
            for i in range(self.num_layers):
                layer = self.gpt_neox.layers[i]
                hidden_states = layer(hidden_states, position_embeddings=position_embeddings)[0]
                # Now, let us do the quantization.
                if i == layer_of_interest and ratio > 0:
                    # Get the ratio amount of the least important token positions
                    least_important_token_positions = torch.argsort(importance_values, descending=False)[
                                                      :int(ratio * batched_input_tokens.size(1))]
                    # Quantize the hidden state activations corresponding to these token positions.
                    # Also, need the shape of the hidden states.
                    # Batch x seq x hidden_size
                    # Simulate symmetric 4-bit integer quantization
                    # Determine the maximum absolute value for scaling
                    max_val = torch.max(torch.abs(hidden_states[:, least_important_token_positions, :]))
                    # Scale the values to the range of 4-bit signed integers (-8 to 7)
                    # Number of levels is 2^4 = 16. For symmetric, we use range -2^(bits-1) to 2^(bits-1)-1
                    num_levels = 16
                    scaled_values = torch.clamp(
                        hidden_states[:, least_important_token_positions, :] / max_val * (num_levels / 2 - 1),
                        -(num_levels / 2), (num_levels / 2 - 1))
                    # Round to the nearest integer
                    quantized_values = torch.round(scaled_values)
                    # Scale back to the original range
                    dequantized_hidden_states = quantized_values / (num_levels / 2 - 1) * max_val

                    hidden_states[:, least_important_token_positions, :] = dequantized_hidden_states

                    ### NOTE: THIS IS A SIMPLIFIED SIMULATION OF SYMMETRIC INT4 QUANTIZATION.
                    ### REAL QUANTIZATION INVOLVES MORE NUANCES.

            post_norm = self.final_layer_norm(hidden_states)
            logits = self.model.embed_out(post_norm)
            # logits shape: (batch_size, seq_len, vocab_size)
            # targets are just the inputs. so, there is a need to shift them by 1
            target = target[:, 1:]
            # Since we do not have targets for the last token, we need to shift those as well
            logits = logits[:, :-1, :]

            # Calculate Cross entropy loss, which is the NLL in perplexity calculation
            cross_entropy = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1),
                                                              ignore_index=- 100)

        # Return the scalar negative log likelihood
        return cross_entropy
