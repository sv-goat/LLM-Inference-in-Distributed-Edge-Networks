import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc


class Pythia_model:
    def __init__(self, model_path, tokenizer_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.model.eval()
        self.device = device

    def tokenize(self, text):
        # print("getting inputs ids")
        return self.tokenizer(text, return_tensors="pt").input_ids

    # Try this on notebook before you commit to this
    def forward(self, inputs):
        # print("forwarding")
        # Send the inputs to the model, and get the inbetween activations
        output = self.model(**inputs, output_attentions = True)
        attentions = output.attentions
        return attentions