import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenPointFiveBModel:
    """
    A layer-wise implementation of the Qwen2-0.5B model for quantization experiments.

    This class provides fine-grained control over the model's forward pass, allowing for
    custom processing of attention mechanisms and intermediate activations. It's designed
    for quantization experiments and layer-wise analysis.

    Args:
        device (str): The device to run the model on ('cuda' or 'cpu')
    """
    def __init__(self, device) -> None:
        self.model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-0.5B', attn_implementation = "sdpa")
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B')
        self.tokenizer = tokenizer
        self.model.to(device)
        self.device = device
        self.num_layers = len(self.model.base_model.layers)
        self.qwen = self.model.base_model
        self.final_layer_norm = self.model.model.norm
        self.rotary_emb = self.qwen.rotary_emb


    def last_portion(self, post_norm, target):
        logits = self.model.lm_head(post_norm)
        # logits shape: (batch_size, seq_len, vocab_size)
        # targets are just the inputs. so, there is a need to shift them by 1
        target = target[:, 1:]
        # Since we do not have targets for the last token, we need to shift those as well
        logits = logits[:, :-1, :]

        # Calculate Cross entropy loss, which is the NLL in perplexity calculation
        cross_entropy = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)),target.view(-1), ignore_index=- 100)

        # Return the scalar negative log likelihood
        return cross_entropy

    def activation_quantization(self, batched_input_tokens, target, importance_values, ratio, layer_of_interest):
        batched_input_tokens = batched_input_tokens.to(self.device)

        with torch.no_grad():
            hidden_states = self.qwen.embed_tokens(batched_input_tokens)
            position_ids = torch.arange(batched_input_tokens.size(1), dtype=torch.long, device=batched_input_tokens.device).unsqueeze(0).expand(batched_input_tokens.size(0), -1)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

            for i in range(self.num_layers):
                layer = self.qwen.layers[i]
                hidden_states = layer(hidden_states,position_embeddings=position_embeddings)[0]
                # Quantization portion
                if i == layer_of_interest and ratio > 0:
                  # Get the ratio amount of the least important token positions.
                  # Eg, if ratio is 0.1, we get the 10% least important tokens
                  least_important_token_positions = torch.argsort(importance_values, descending=False)[:int(ratio * batched_input_tokens.size(1))]
                  # Simulate symmetric 4-bit integer quantization
                  # Determine the maximum absolute value for scaling
                  max_val = torch.max(torch.abs(hidden_states[:, least_important_token_positions, :]))
                  # Scale the values to the range of 4-bit signed integers (-8 to 7)
                  # Number of levels is 2^4 = 16. For symmetric, we use range -2^(bits-1) to 2^(bits-1)-1
                  num_levels = 16
                  scaled_values = torch.clamp(hidden_states[:, least_important_token_positions, :] / max_val * (num_levels / 2 - 1), -(num_levels / 2), (num_levels / 2 - 1))
                  # Round to the nearest integer
                  quantized_values = torch.round(scaled_values)
                  # Scale back to the original range
                  dequantized_hidden_states = quantized_values / (num_levels / 2 - 1) * max_val

                  hidden_states[:, least_important_token_positions, :] = dequantized_hidden_states

                  ### NOTE: THIS IS A SIMPLIFIED SIMULATION OF SYMMETRIC INT4 QUANTIZATION.
                  ### REAL QUANTIZATION INVOLVES MORE NUANCES.

            post_norm = self.final_layer_norm(hidden_states)
            return self.last_portion(post_norm, target)

    def layer_by_layer_impl(self, batched_input_tokens, target):
          # input ids have shape batch size x num tokens
          batched_input_tokens = batched_input_tokens.to(self.device)
          # Get the final layer norm used in qwen architecture
          final_layer_norm = self.model.model.norm

          qwen = self.model.base_model
          rotary_emb = qwen.rotary_emb

          with torch.no_grad():
              hidden_states = qwen.embed_tokens(batched_input_tokens)
              # hidden_states shape: (batch_size, seq_len, model_dim)
              # Create position embeddings of shape batch_size, seq _len
              # Value of the position embeddings should be between 0 and the number of positions -1
              position_ids = torch.arange(batched_input_tokens.size(1), dtype=torch.long,
                                          device=batched_input_tokens.device).unsqueeze(0).expand(
                  batched_input_tokens.size(0), -1)
              position_embeddings = rotary_emb(hidden_states, position_ids)

              for i in range(self.num_layers):
                  layer = qwen.layers[i]
                  hidden_states = layer(hidden_states, position_embeddings=position_embeddings)[0]
                  # Shape
                  # ( batch, num_input_tokens, model_dim )

              post_norm = final_layer_norm(hidden_states)
              return self.last_portion(post_norm, target)

    def channel_wise(self, batched_input_tokens, target, layer_of_interest, method):
          batched_input_tokens = batched_input_tokens.to(self.device)
          if method == "channel_8":
            max_levels = 127
          elif method == "channel_4":
            max_levels = 7

          with torch.no_grad():
              hidden_states = self.qwen.embed_tokens(batched_input_tokens)
              position_ids = torch.arange(batched_input_tokens.size(1), dtype=torch.long, device=batched_input_tokens.device).unsqueeze(0).expand(batched_input_tokens.size(0), -1)
              position_embeddings = self.rotary_emb(hidden_states, position_ids)
              # How many layers
              for i in range(self.num_layers):
                  layer = self.qwen.layers[i]
                  hidden_states = layer(hidden_states,position_embeddings=position_embeddings)[0]
                  # Now, let us do the quantization.
                  if i == layer_of_interest:
                    # Quantize channel wise
                    # Iterate over the channel dimension
                    for c in range(hidden_states.shape[2]):
                        # Get the hidden states for the current channel
                        channel_hidden_states = hidden_states[:, :, c]
                        if method == "channel_8" or method == "channel_4":
                          # Get the max value
                          channel_max = torch.max(torch.abs(channel_hidden_states))
                          # quantize to int8
                          quantized_channel_hidden_states = torch.round(channel_hidden_states / channel_max * max_levels)
                          # dequantize back to fp32
                          dequantized_channel_hidden_states = quantized_channel_hidden_states * channel_max / max_levels
                        elif method == "channel_1_mean":
                          # Get the mean value and add a value to ensure numerical stability.
                          channel_mean = torch.mean(channel_hidden_states) + 1e-8
                          # round clip to {-1, 0, 1}
                          quantized_channel_hidden_states = torch.round(channel_hidden_states / channel_mean)
                          quantized_channel_hidden_states = torch.clamp(quantized_channel_hidden_states, -1, 1)
                          # dequantize back to fp32
                          dequantized_channel_hidden_states = quantized_channel_hidden_states * channel_mean
                        elif method == "channel_1_max":
                          # Get the max value
                          channel_max = torch.max(torch.abs(channel_hidden_states))
                          # round clip to {-1, 0, 1}
                          quantized_channel_hidden_states = torch.round(channel_hidden_states / channel_max)
                          quantized_channel_hidden_states = torch.clamp(quantized_channel_hidden_states, -1, 1)
                          # dequantize back to fp32
                          dequantized_channel_hidden_states = quantized_channel_hidden_states * channel_max
                        # Replace the hidden states for the current channel with the quantized values
                        hidden_states[:, :, c] = dequantized_channel_hidden_states


              post_norm = self.final_layer_norm(hidden_states)
              logits = self.model.lm_head(post_norm)
              # logits shape: (batch_size, seq_len, vocab_size)
              # targets are just the inputs. so, there is a need to shift them by 1
              target = target[:, 1:]
              # Since we do not have targets for the last token, we need to shift those as well
              logits = logits[:, :-1, :]

              # Calculate Cross entropy loss, which is the NLL in perplexity calculation
              cross_entropy = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)),target.view(-1), ignore_index=- 100)

          # Return the scalar negative log likelihood
          return cross_entropy
