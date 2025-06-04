import os
import subprocess
from concurrent.futures import as_completed, ThreadPoolExecutor

# Define model choices and data buckets
model_choices = ["4bit", "8bit", "default"]
buckets = ["XS", "S", "M", "L", "XL", "XXL"]


def run_task(model_type, bucket_size, gpu_id):
    """Run a single task on the specified GPU."""
    try:
        command = [
            "python",
            "execute.py",
            "--bucket_size",
            bucket_size,
            "--model_type",
            model_type,
            "--device",
            str(gpu_id),
        ]
        print(f"Running {model_type} on bucket {bucket_size} using GPU {gpu_id}")
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(
            f"Error running {model_type} on bucket {bucket_size} using GPU {gpu_id}: {e}"
        )


def main():
    # Create a list of tasks
    tasks = [(model, bucket) for model in model_choices for bucket in buckets]

    # Create a thread pool executor with 8 workers (one per GPU)
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit initial tasks to each GPU
        futures = {
            executor.submit(run_task, model, bucket, gpu_id): gpu_id
            for gpu_id, (model, bucket) in enumerate(tasks[:8])
        }

        # Process remaining tasks as GPUs become available
        task_iter = iter(tasks[8:])
        while futures:
            done_future = next(as_completed(futures))
            gpu_id = futures.pop(done_future)

            try:
                model, bucket = next(task_iter)
                future = executor.submit(run_task, model, bucket, gpu_id)
                futures[future] = gpu_id
            except StopIteration:
                # No more tasks to process
                pass


if __name__ == "__main__":
    main()
