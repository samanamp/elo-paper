import csv
import os
import re


def parse_filename(filename):
    # Split the filename into components
    parts = filename.split("_")
    judge_model = parts[0]
    target_model = parts[1]
    ref_model = parts[2]
    bucket_size = parts[3]
    gen_tokens = parts[4]
    num_prompts = parts[5]
    return judge_model, target_model, ref_model, bucket_size, gen_tokens, num_prompts


def parse_file_content(content):
    # Use regex to extract the required values from the content
    match = re.search(
        r"target valid (\d+\.\d+)% win: (\d+\.\d+)% ref valid (\d+\.\d+)% win: (\d+\.\d+)% .* Elo delta: (-?\d+\.\d+).*?(\d+)=(\d+)",
        content,
    )
    if match:
        target_valid = match.group(1)
        target_win = match.group(2)
        ref_valid = match.group(3)
        ref_win = match.group(4)
        elo_delta = match.group(5)
        valid_responses = match.group(6)
        total_responses = match.group(7)
        return (
            target_valid,
            target_win,
            ref_valid,
            ref_win,
            elo_delta,
            valid_responses,
            total_responses,
        )
    return None


def process_files(directory):
    output_data = []

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as file:
                content = file.read()
                parsed_filename = parse_filename(filename)
                parsed_content = parse_file_content(content)

                if parsed_content:
                    output_data.append(parsed_filename + parsed_content)

    return output_data


def write_to_csv(data, output_file):
    headers = [
        "judge_model",
        "target_model",
        "target_valid",
        "target_win",
        "ref_model",
        "ref_valid",
        "ref_win",
        "bucket_size",
        "gen_tokens",
        "num_prompts",
        "elo_delta",
        "valid_responses",
        "total_responses",
    ]

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(data)


if __name__ == "__main__":
    directory = "/tmp/samanamp/elo5"  # Change this to your directory path
    output_file = "output-long.csv"
    data = process_files(directory)
    write_to_csv(data, output_file)
    print(f"Data has been written to {output_file}")
