import json
from collections import defaultdict

from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset
dataset = load_dataset("lmsys/lmsys-chat-1m", split="train[:100%]")

# Load Gemma-3 tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

# Define token buckets
buckets = {
    "XS": (0, 128),
    "S": (129, 256),
    "M": (257, 512),
    "L": (513, 1024),
    "XL": (1025, 2048),
    "XXL": (2049, 4096),
    # "XXXL": (10000, 20000),
}

# Prepare bucketed conversations
bucketed_data = defaultdict(list)


def get_bucket(token_len):
    for bucket, (low, high) in buckets.items():
        if low <= token_len <= high:
            return bucket
    return None  # Skip if longer than highest bucket


bucket_count = defaultdict(int)
# bucket_count["default"] += 1
MAX_COUNT = 1000
item_count = 0
# Iterate through dataset
for convo in dataset:
    item_count += 1
    # print(convo)
    messages = convo["conversation"]

    # Remove last assistant message if it's from assistant
    if messages and messages[-1]["role"] == "assistant":
        messages = messages[:-1]

    # Convert messages to plain text (e.g., format as chat)
    prompt = ""
    for m in messages:
        role = m["role"]
        content = m["content"]
        prompt += f"{role}: {content}\n"

    # Tokenize
    tokenized = tokenizer(prompt, return_tensors="pt", truncation=False)
    token_len = tokenized.input_ids.shape[-1]

    # Bucketize
    bucket = get_bucket(token_len)
    if bucket:
        if bucket_count[bucket] > MAX_COUNT:
            continue
        bucketed_data[bucket].append(
            {
                "prompt": messages,
                "token_count": token_len,
                "conversation_id": convo["conversation_id"],
            }
        )
        bucket_count[bucket] += 1
    if all(count > MAX_COUNT for count in bucket_count.values()):
        print("All items are greater than MAX_COUNT. Breaking the loop.")
        break
    print(item_count, "::", bucket_count)

# (Optional) Save to disk
# with open("bucketed_conversations.json", "w") as f:
#     json.dump(bucketed_data, f, indent=2)

for bucket, data in bucketed_data.items():
    filename = f"bucket_{bucket}_1k.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {bucket} bucket to {filename}")

print("Done. Buckets and prompts saved.")
