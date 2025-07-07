# Requisite imports
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# I want to pass through the model layer by layer.
class Pythia70Model:
    def __init__(self, device, ratios) -> None:
        self.model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-70m-deduped')
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m-deduped')
        self.tokenizer = tokenizer
        self.model.to(device)
        self.device = device
        self.ratios = ratios
        self.final_layer_norm = self.model.gpt_neox.final_layer_norm

    def calculate_batched_nll(self, hidden_states, target):
        with torch.no_grad():
            post_norm = self.final_layer_norm(hidden_states)
            logits = self.model.embed_out(post_norm)
            # logits shape: (batch_size, seq_len, vocab_size)
            # targets are just the inputs. so, there is a need to shift them by 1
            target = target[:, 1:]
            # Since we do not have targets for the last token, we need to shift those as well
            logits = logits[:, :-1, :]

            # I want seperate nll for each of the batches
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

    def batched_top_rho_quantization(self, batched_input_tokens, distribution, target, quant_layer=2, do_quant=True):
        # input ids have shape batch size x num tokens
        batched_input_tokens = batched_input_tokens.to(self.device)
        # Get the model
        gpt_neox = self.model.base_model
        rotary_emb = gpt_neox.rotary_emb

        with torch.no_grad():
            # Get the hidden states
            hidden_states = gpt_neox.embed_in(batched_input_tokens)
            # Create position embeddings of shape batch_size , seq _len
            # Value of the position embeddings should be between 0 and the number of positions -1
            position_ids = torch.arange(batched_input_tokens.size(1), dtype=torch.long,
                                        device=batched_input_tokens.device).unsqueeze(0).expand(
                batched_input_tokens.size(0), -1)
            position_embeddings = rotary_emb(hidden_states, position_ids)
            # How many layers
            for i in range(6):
                layer = gpt_neox.layers[i]
                hidden_states = layer(hidden_states, position_embeddings=position_embeddings)[0]
                # What is the shape of this hidden states?
                # ( batch, num_input_tokens, model_dim = 512 )

                if i == quant_layer and do_quant:
                    # I'll only quantize the activations according to some measure of importance
                    # ( batch, num_input_tokens, model_dim = 512 )
                    for ratio in self.ratios:
                        # First, let us define the threshold that we want to reach.
                        threshold = 1 - 0.1 * ratio
                        # Now, we want to choose the fewest indices that can add upto this threshold from the given distribution
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

                        # Now that we have the tokens to quantize, we just simply have to follow the other function from here.
                        hidden_states = self.simulate_quantization(tokens_to_quantize, hidden_states, ratio)

            return self.calculate_batched_nll( hidden_states, target)

    def batched_quantization_helper(self, batched_input_tokens, ordering, target, quant_layer=2, do_quant=True):
        # input ids have shape batch size x num tokens
        batched_input_tokens = batched_input_tokens.to(self.device)
        # Get the model
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
            # How many layers
            for i in range(6):
                layer = gpt_neox.layers[i]
                hidden_states = layer(hidden_states, position_embeddings=position_embeddings)[0]
                # What is the shape of this hidden states?
                # ( batch, num_input_tokens, model_dim = 512 )

                if i == quant_layer and do_quant:
                    # I'll only quantize the activations according to some measure of importance
                    # ( batch, num_input_tokens, model_dim = 512 )
                    for ratio in self.ratios:
                        tokens_to_quantize = ordering[:int(0.1 * ratio * len(ordering))]
                        # Accumulate the negative log likelihoods

                        hidden_states = self.simulate_quantization(tokens_to_quantize, hidden_states, ratio)

            return self.calculate_batched_nll( hidden_states, target)
