# Requisite imports
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# I want to pass through the model layer by layer.


class Pythia70Model:
    def __init__(self, device = "cuda") -> None:
        self.model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-70m-deduped')
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m-deduped')
        self.tokenizer = tokenizer
        self.model.to(device)
        self.device = device
        self.ratios = [r for r in range(10)]

    def quantization_helper(self, input_tokens, tokens_to_quantize, target, quant_layer = 2, do_quant = True ):
        # input ids have shape batch size x num tokens
        input_tokens = input_tokens.to(self.device)
        # Get the model
        final_layer_norm = self.model.gpt_neox.final_layer_norm
        gpt_neox = self.model.base_model
        rotary_emb = gpt_neox.rotary_emb

        # Autoregressively generate new token
        with torch.no_grad():
            # Get the hidden states
            hidden_states = gpt_neox.embed_in(input_tokens)
            position_ids = torch.arange(input_tokens.size(1), dtype=torch.long, device=input_tokens.device).unsqueeze(0)
            position_embeddings = rotary_emb(hidden_states, position_ids)
            # How many layers
            for i in range(6):
                layer = gpt_neox.layers[i]
                hidden_states = layer(hidden_states, position_embeddings=position_embeddings)[0]
                # What is the shape of this hidden state variable?
                # ( batch, num_input_tokens, model_dim = 512 )

                if i == quant_layer and do_quant:
                    # I'll only quantize the activations according to some measure of importance
                    # ( batch, num_input_tokens, model_dim = 512 )
                    for token in tokens_to_quantize:
                      activation_for_token = hidden_states[:, token, :]
                      scale = torch.max(activation_for_token) - torch.min(activation_for_token)
                      zero_point = torch.min(activation_for_token)
                      quantized_activation_for_token = torch.quantize_per_tensor(activation_for_token, scale, zero_point, torch.qint8)
                      unquantized_activation_for_token = torch.dequantize(quantized_activation_for_token)
                      # Replace the activations
                      hidden_states[:, token, :] = unquantized_activation_for_token


            post_norm = final_layer_norm(hidden_states)
            logits = self.model.embed_out(post_norm)
            # logits shape: (batch_size, seq_len, vocab_size)

            # We want the logits for the prediction of the next token,
            # which are the logits at the last position of the input sequence.
            next_token_logits = logits[:, -1, :] # Shape: (batch_size, vocab_size)

            # Since we are predicting a single target token, the target
            # should also be a tensor with shape (batch_size,).
            # Assuming batch_size is 1 and target is an integer token ID:
            target_tensor = torch.tensor([target], dtype=torch.long, device=self.device)

            # Calculate the Cross-Entropy Loss (which is equivalent to NLL
            # for a single target).
            # cross_entropy expects logits (unnormalized scores) and target indices.
            # The reduction='none' gives the loss per sample in the batch.
            loss = torch.nn.functional.cross_entropy(next_token_logits, target_tensor, reduction='none')

            # Since we assume batch size 1, the loss for this prediction is the NLL.
            nll = loss.squeeze(0) # Remove batch dimension
            if nll > 100:
              print("nll " + str(nll))


        # Return the scalar negative log likelihood
        return nll

    def batched_quantization_helper(self, batched_input_tokens, ordering, target, quant_layer = 2, do_quant = True ):
        # input ids have shape batch size x num tokens
        batched_input_tokens = batched_input_tokens.to(self.device)
        # Get the model
        final_layer_norm = self.model.gpt_neox.final_layer_norm
        gpt_neox = self.model.base_model
        rotary_emb = gpt_neox.rotary_emb


        # Autoregressively generate new token
        with torch.no_grad():
            # Get the hidden states
            hidden_states = gpt_neox.embed_in(batched_input_tokens)
            # Create position embeddings of shape batch_size , seq _len
            # Value of the position embeddings should be between 0 and the number of positions -1
            position_ids = torch.arange(batched_input_tokens.size(1), dtype=torch.long, device=batched_input_tokens.device).unsqueeze(0).expand(batched_input_tokens.size(0), -1)
            position_embeddings = rotary_emb(hidden_states, position_ids)
            # How many layers
            for i in range(6):
                layer = gpt_neox.layers[i]
                hidden_states = layer(hidden_states, position_embeddings=position_embeddings)[0]
                # What is the shape of this hidden state?
                # ( batch, num_input_tokens, model_dim = 512 )

                if i == quant_layer and do_quant:
                    # I'll only quantize the activations according to some measure of importance
                    # ( batch, num_input_tokens, model_dim = 512 )
                    for ratio in self.ratios:
                      tokens_to_quantize = ordering[:int(0.1 * ratio * len(ordering))]
                      # Accumulate the negative log likelihoods

                      for token in tokens_to_quantize:
                        activation_for_token = hidden_states[ratio, token, :]
                        scale = torch.max(activation_for_token) - torch.min(activation_for_token)
                        zero_point = torch.min(activation_for_token)
                        quantized_activation_for_token = torch.quantize_per_tensor(activation_for_token, scale, zero_point, torch.qint8)
                        unquantized_activation_for_token = torch.dequantize(quantized_activation_for_token)
                        # Replace the activations
                        hidden_states[ratio, token, :] = unquantized_activation_for_token


            post_norm = final_layer_norm(hidden_states)
            logits = self.model.embed_out(post_norm)
            # logits shape: (batch_size, seq_len, vocab_size)

            # We want the logits for the prediction of the next token,
            # which are the logits at the last position of the input sequence.
            next_token_logits = logits[:, -1, :] # Shape: (batch_size, vocab_size)

            # Since we are predicting a single target token, the target
            # should also be a tensor with shape (batch_size,).
            # Assuming batch_size is 1 and target is an integer token ID:
            target_tensor = torch.tensor([target] * 10, dtype=torch.long, device=self.device)

            # Calculate the Cross-Entropy Loss (which is equivalent to NLL
            # for a single target).
            # cross_entropy expects logits (unnormalized scores) and target indices.
            # The reduction='none' gives the loss per sample in the batch.
            loss = torch.nn.functional.cross_entropy(next_token_logits, target_tensor, reduction='none')

            # Since we assume batch size 1, the loss for this prediction is the NLL.
            batched_nll = loss


        # Return the scalar negative log likelihood
        return batched_nll