from transformers import AutoTokenizer
import pickle
from tqdm import tqdm
from datasets import load_dataset
import math
from qwen_layer_wise import QwenPointFiveBModel
from transformers import AutoModelForCausalLM


def channel_wise(layers_of_interest, methods):
    wikitext = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B')

    # Join the entire corpus
    encodings = tokenizer("\n\n".join(wikitext["text"]), return_tensors="pt")

    device = "cuda"
    qwen_model = QwenPointFiveBModel(device)

    enthu_qwen = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-0.5B', attn_implementation="eager")
    enthu_qwen.eval()
    enthu_qwen.to(device)

    max_length = 512
    stride = 32
    seq_len = encodings.input_ids.size(1)
    prev_end_loc = 0

    total_nll = [[0 for __ in range(len(methods))] for _ in range(len(layers_of_interest))]
    n_tokens = 0

    iterations = 0

    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift

        for m, method in enumerate(methods):
            for l, layer_of_interest in enumerate(layers_of_interest):
                neg_log_likelihood = qwen_model.channel_wise(input_ids, target_ids, layer_of_interest, method)
                total_nll[l][m] += (neg_log_likelihood.item() * num_loss_tokens)

        n_tokens += num_loss_tokens

        prev_end_loc = end_loc

        if iterations % 1000 == 0:
            print(f"Processed {iterations} chunks")
            print(f"Total NLL: {total_nll}")
            print(f"Total tokens: {n_tokens}")
            # Save the total_nll and n_tokens to a file
            with open('total_nll_channel.pkl', 'wb') as f:
                pickle.dump(total_nll, f)
            with open('n_tokens_channel.pkl', 'wb') as f:
                pickle.dump(n_tokens, f)

        iterations += 1

        if end_loc == seq_len:
            break

        avg_ppl_results = [[0 for __ in range(len(methods))] for _ in range(len(layers_of_interest))]
        for m, method in enumerate(methods):
            for l, layer_of_interest in enumerate(layers_of_interest):
                avg_ppl_results[l][m] = math.exp(total_nll[l][m] / n_tokens)

        print(avg_ppl_results)

        with open('ppl_channel.pkl', 'wb') as f:
            pickle.dump(avg_ppl_results, f)
