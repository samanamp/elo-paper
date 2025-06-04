import torch

# import torch.distributed as dist
# import torch.multiprocessing as mp
# from datasets import load_dataset
from transformers import (
    # AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Gemma3ForCausalLM,
    GPTQConfig,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
quantization_bits = 8
model_id = "google/gemma-3-1b-it"
quantized_model_name = f"gemma-3-1b-it-gptq-{quantization_bits}bit"

tokenizer = AutoTokenizer.from_pretrained(model_id)

quantization_config = GPTQConfig(
    bits=quantization_bits,
    dataset="c4",
    tokenizer=tokenizer,
    damp_percent=0.01,  # Example of adjusting damp percent
    block_size=128,
)

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
quantization_config = BitsAndBytesConfig(load_in_8bit=True)


# Move the model to the specified device
quantized_model = Gemma3ForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    attn_implementation="eager",
    # max_memory={0: "60GiB", 1: "60GiB", "cpu": "60GiB"},
    # quantization_config=GPTQConfig(bits=4, backend="marlin", dataset="c4"),
    quantization_config=quantization_config,
)

quantized_model.save_pretrained(quantized_model_name)
tokenizer.save_pretrained(quantized_model_name)
