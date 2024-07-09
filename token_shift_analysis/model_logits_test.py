import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    inputs = ["I went", "I went for", "I went for a", "I went for a walk in the park"]
    probability_distribution_all = []
    model_name = "stabilityai/stablelm-2-1_6b"
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True)
    max_len = 1024
    truncation = True

    last_logits = []

    for input in inputs:
        # Tokenize input text
        input_ids = tokenizer.encode(input, return_tensors="pt", max_length=max_len, truncation=truncation).to("cuda")

        # Generate probabilities for the next token
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]  # Take the logits for the last token
            last_logits.append(logits)
            # probabilities = torch.nn.functional.softmax(logits, dim=-1)
            # probability_distribution_all.append(probabilities)
    token_lengths = [len(tokenizer.encode(input, add_special_tokens=True)) - 1 for input in
                     inputs[:-1]]  # Adjusted for special tokens
    for index, token_length in enumerate(token_lengths):
        single_logit = last_logits[index]
        final_logits = outputs.logits[:, token_length, :]
        assert torch.allclose(single_logit, final_logits, atol=1e-5)
    print("Passed test")


if __name__ == "__main__":
    main()
