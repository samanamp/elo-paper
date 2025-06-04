import argparse
import json

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, Gemma3ForCausalLM


def adapt_messages(messages):
    """Convert messages from JSON format to the format expected by the model."""
    adapted_messages = []
    for message in messages:
        adapted_message = {
            "role": message["role"],
            "content": [{"type": "text", "text": message["content"]}],
        }
        adapted_messages.append(adapted_message)
    return [adapted_messages]  # Wrap in a list to match the expected input format


def extract_model_output(decoded_output):
    """Extract the model's output from the decoded text."""
    # Assuming the model's output starts after the last user message
    # This is a simple heuristic; adjust as needed based on actual output format
    output_start = decoded_output.find("model\n") + len("model\n")
    return decoded_output[output_start:].strip()


def main(bucket_size, model_type, device_id):
    # Set device
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    # Model type mapping
    model_types = {
        "2bit": "gemma-3-1b-it-gptq-2bit",
        "4bit": "gemma-3-1b-it-gptq-4bit",
        "8bit": "gemma-3-1b-it-gptq-8bit",
        "default": "google/gemma-3-1b-it",
    }
    model_id = model_types.get(model_type, "default")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    quantized_model = Gemma3ForCausalLM.from_pretrained(
        model_id,
        attn_implementation="eager",
        device_map={"": device},
    ).eval()

    # Load bucket file
    bucket_filename = f"bucket_{bucket_size}.json"
    with open(bucket_filename, "r") as f:
        bucket_data = json.load(f)

    # Prepare results list
    results = []

    # Iterate over entries with a progress bar
    for entry in tqdm(bucket_data, desc="Processing entries", unit="entry"):
        # Extract and adapt prompt
        messages = entry["prompt"]
        adapted_messages = adapt_messages(messages)

        # Tokenize and prepare inputs
        inputs = tokenizer.apply_chat_template(
            adapted_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        # Generate output
        with torch.inference_mode():
            outputs = quantized_model.generate(**inputs, max_new_tokens=512)

        # Decode and extract model's output
        decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        model_output = extract_model_output(decoded_output)

        # Append result to list
        results.append(
            {"conversation_id": entry["conversation_id"], "model_output": model_output}
        )

    # Write results to JSON file
    output_filename = f"output_bucket_{bucket_size}_model_{model_type}.json"
    with open(output_filename, "w") as output_file:
        json.dump(results, output_file, indent=2)

    print(f"Done. Outputs saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model with specified parameters.")
    parser.add_argument(
        "--bucket_size",
        type=str,
        required=True,
        help="Bucket size for processing (e.g., 'L').",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["2bit", "4bit", "8bit", "default"],
        help="Model type to use.",
    )
    parser.add_argument(
        "--device", type=int, required=True, help="CUDA device ID to use."
    )

    args = parser.parse_args()
    main(args.bucket_size, args.model_type, args.device)
