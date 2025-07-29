from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class QwenPointFiveBModel:
    def __init__(self, device) -> None:
        self.model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-0.5B')
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B')
        self.tokenizer = tokenizer
        self.model.to(device)
        self.device = device
        self.num_layers = len(self.model.base_model.layers)

        self.final_layer_norm = self.model.model.norm
        self.qwen = self.model.base_model
        self.rotary_emb = self.qwen.rotary_emb

    def activation_quantization(self, batched_input_tokens, target, importance_values, ratio, layer_of_interest):
        batched_input_tokens = batched_input_tokens.to(self.device)

        with torch.inference_mode():
            hidden_states = self.qwen.embed_tokens(batched_input_tokens)
            position_ids = torch.arange(batched_input_tokens.size(1), dtype=torch.long, device=batched_input_tokens.device).unsqueeze(0).expand(batched_input_tokens.size(0), -1)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            # How many layers
            for i in range(self.num_layers):
                layer = self.qwen.layers[i]
                hidden_states = layer(hidden_states,position_embeddings=position_embeddings)[0]
                # Now, let us do the quantization.
                if i == layer_of_interest:
                  # Get the ratio amount of the least important token positions
                  least_important_token_positions = torch.argsort(importance_values, descending=False)[:int(ratio * batched_input_tokens.size(1))]
                  # Quantize the hidden state activations corresponding to these token positions.
                  # Also, need the shape of the hidden states.
                  # Batch x seq x hidden_size
                  # Simulate simple quantization by rounding and scaling
                  quantized_hidden_states = torch.round(hidden_states[:, least_important_token_positions, :] * 255.0) / 255.0
                  hidden_states[:, least_important_token_positions, :] = quantized_hidden_states

                  ### NOTE: THIS IS VERY SIMPLE QUANT. FOR REALISTIC SIM, MAYBE
                  ### USE LIBS??

            return self.final_layer_shens(hidden_states, target)


    def final_layer_shens(self, hidden_states, target):
        with torch.inference_mode():
            post_norm = self.final_layer_norm(hidden_states)
            logits = self.model.lm_head(post_norm)
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

    def layer_by_layer_impl(self, batched_input_tokens, target):
        # input ids have shape batch size x num tokens
        batched_input_tokens = batched_input_tokens.to(self.device)
        # Get the final layer norm used in qwen architecture
        final_layer_norm = self.model.model.norm
        # Get the model
        qwen = self.model.base_model
        rotary_emb = qwen.rotary_emb

        with torch.inference_mode():
            hidden_states = qwen.embed_tokens(batched_input_tokens)
            # hidden_states shape: (batch_size, seq_len, model_dim
            # Create position embeddings of shape batch_size , seq _len
            # Value of the position embeddings should be between 0 and the number of positions -1
            position_ids = torch.arange(batched_input_tokens.size(1), dtype=torch.long,
                                        device=batched_input_tokens.device).unsqueeze(0).expand(
                batched_input_tokens.size(0), -1)
            position_embeddings = rotary_emb(hidden_states, position_ids)
            # How many layers
            for i in range(self.num_layers):
                layer = qwen.layers[i]
                hidden_states = layer(hidden_states, position_embeddings=position_embeddings)[0]
                # What is the shape of this hidden states?
                # ( batch, num_input_tokens, model_dim )

            return self.final_layer_shens(hidden_states, target)