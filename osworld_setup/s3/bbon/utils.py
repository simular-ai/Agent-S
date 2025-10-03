import logging
import os
import re
import json
from PIL import Image
from typing import Optional, List
import base64


def image_to_openai_message_format(
    image_path: str, caption: str = None
) -> Optional[dict]:
    """Convert an image file to OpenAI message format."""
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return None

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        if not image_bytes:
            print(f"Empty image file: {image_path}")
            return None

        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        if not base64_image:
            print(f"Failed to encode image to base64: {image_path}")
            return None

        content = []
        if caption:
            content.append({"type": "text", "text": caption})

        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            }
        )

        return {"role": "user", "content": content}

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def load_facts(task_dir: str) -> List[str]:
    """Load existing facts from facts.jsonl file."""
    fact_captions_file = os.path.join(task_dir, "fact_captions.jsonl")

    if not os.path.exists(fact_captions_file):
        print(f"fact_captions.jsonl not found at {fact_captions_file}")
        return []

    fact_captions = []
    with open(fact_captions_file, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if "fact_answer" in data:
                    fact_captions.append(data["fact_answer"])

    return fact_captions


def load_task_instruction(task: str, examples_path: str) -> Optional[str]:
    """
    Load task instruction from examples path.

    Args:
        task: Task ID in format "domain/example_id"
        examples_path: Path to the examples directory (e.g., "/home/ubuntu/Simular/OSWorld/evaluation_examples/examples")

    Returns:
        Task instruction string or None if not found
    """
    domain, example_id = task.split("/", 1)

    # Construct path to the JSON file
    json_file_path = os.path.join(examples_path, domain, f"{example_id}.json")

    if not os.path.exists(json_file_path):
        logging.warning(f"Example file not found: {json_file_path}")
        return None

    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract instruction from the JSON
        if "instruction" in data:
            instruction = data["instruction"]
            if instruction and instruction.strip():
                return instruction.strip()

        logging.warning(f"No 'instruction' key found in {json_file_path}")
        return None

    except Exception as e:
        logging.warning(f"Error reading example file {json_file_path}: {e}")
        return None


def get_final_screenshot_file(result_dir: str) -> str:
    """
    Finds the screenshot file with the largest valid step index in the given directory.
    Works with filenames like step_0.png, step_1_20250.png, step-2.png, etc.
    Only considers .png files (case-insensitive).
    If the highest index file is invalid/corrupted, it tries the next lower index.
    Returns None if no valid matching files are found.
    """
    # First, collect all valid step files with their indices
    step_files = {}
    pattern = re.compile(r"step[_\-]?(\d+)", re.IGNORECASE)

    for fname in os.listdir(result_dir):
        if not fname.lower().endswith(".png"):
            continue
        match = pattern.match(fname)
        if match:
            idx = int(match.group(1))
            step_files[idx] = fname
    if not step_files:
        return None
    # Sort indices in descending order (highest first)
    sorted_indices = sorted(step_files.keys(), reverse=True)
    # Try each file from highest to lowest index
    for idx in sorted_indices:
        fname = step_files[idx]
        file_path = os.path.join(result_dir, fname)
        # Check if file exists and is valid
        if os.path.exists(file_path) and is_valid_image(file_path):
            return fname
        else:
            print(
                f"Invalid or corrupted image at step {idx}: {fname}, trying previous step..."
            )
    return None


def is_valid_image(file_path: str) -> bool:
    """
    Check if an image file is valid by trying to open it with PIL.
    Also checks if file is not empty.
    """
    try:
        # Check file size first (quick check)
        if os.path.getsize(file_path) == 0:
            return False

        # Try to open and verify the image
        with Image.open(file_path) as img:
            img.verify()  # This will raise an exception if image is corrupted
            return True
    except Exception as e:
        print(f"Image validation failed for {file_path}: {e}")
        return False


def get_new_tasks_classification(results_dirs: [str]):
    # Step 1: collect domain/task_ids for each trajectory
    tasks_per_dir = []
    for results_dir in results_dirs:
        domain_tasks = set()
        for domain in os.listdir(results_dir):
            domain_dir = os.path.join(results_dir, domain)
            if not os.path.isdir(domain_dir):
                continue
            for task_id in os.listdir(domain_dir):
                task_dir = os.path.join(domain_dir, task_id)
                if os.path.isdir(task_dir):
                    domain_tasks.add(f"{domain}/{task_id}")
        tasks_per_dir.append(domain_tasks)

    # Step 2: find tasks common to all trajectories
    common_tasks = set.intersection(*tasks_per_dir)

    constant_tasks = []
    variance_tasks = []
    constant_tasks_scores = []
    optimal_sum = 0.0
    expected_value = 0.0

    # Step 3: evaluate each common task
    for domain_task in sorted(common_tasks):
        domain, task_id = domain_task.split("/", 1)
        results = []
        for results_dir in results_dirs:
            task_dir = os.path.join(results_dir, domain, task_id)
            result_file = os.path.join(task_dir, "result.txt")
            if os.path.isfile(result_file):
                with open(result_file, "r") as f:
                    try:
                        val = float(f.read().strip())
                        results.append(val)
                    except ValueError:
                        continue

        if not results:  # skip if no valid results
            logging.warning(f"No valid results for {domain_task}")
            continue

        # classification
        if all(r == results[0] for r in results):
            constant_tasks.append(domain_task)
            constant_tasks_scores.append(results[0])
        else:
            variance_tasks.append(domain_task)

        # accumulate min/optimal
        # minimum_sum += min(results) #We incorrectly also counted the minimum sum of variance tasks, we should not do this
        optimal_sum += max(results)
        expected_value += sum(results) / len(results)

    return {
        "constant": constant_tasks,  # We dont evaluate constant tasks
        "variance": variance_tasks,  # We evaluate variance tasks
        "minimum": sum(
            constant_tasks_scores
        ),  # sum of constant tasks scores (easy + hard)
        "optimal": optimal_sum,  # If we get the best score, we get the optimal score
        "expected_value": expected_value,  # If we get the average score across all tasks for all trajectories, we get the expected value
    }


def check_selected_trajectory(results_dirs: [str], selected_trajectory: str, task: str):
    """
    results_dirs: list of directories in format results_dir/<domain>/<task_id>
    selected_trajectory: the path of the selected trajectory
    task: string in format "<domain>/<task_id>"

    Returns (selected_val, optimal_val)
    """
    domain, task_id = task.split("/")
    all_results = []

    if not any(
        os.path.commonpath([os.path.abspath(selected_trajectory), os.path.abspath(rd)])
        == os.path.abspath(rd)
        for rd in results_dirs
    ):
        return None, None

    for rd in results_dirs:
        result_file = os.path.join(rd, domain, task_id, "result.txt")
        if os.path.isfile(result_file):
            try:
                all_results.append(float(open(result_file).read().strip()))
            except ValueError:
                pass

    selected_file = os.path.join(selected_trajectory, domain, task_id, "result.txt")
    if not os.path.isfile(selected_file):
        return None, max(all_results) if all_results else None

    try:
        selected_val = float(open(selected_file).read().strip())
    except ValueError:
        return None, max(all_results) if all_results else None

    optimal_val = max(all_results) if all_results else selected_val
    return selected_val, optimal_val


def evaluate_comparative_results(results_dirs: [str], json_path: str = None):
    """
    Opens comparative_judge_results.json (default) or a given path,
    evaluates each task, and returns results.

    Args:
        results_dirs: list of result directories
        json_path: optional path to comparative_judge_results.json

    Returns:
        dict mapping task -> {"selected_val": float or None, "optimal_val": float or None}
    """
    judge_score = 0
    optimal_score = 0
    if json_path is None:
        json_path = "comparative_judge_results.json"

    with open(json_path, "r") as f:
        data = json.load(f)

    results = {}
    for task, info in data.items():
        selected_trajectory = info.get("selected_trajectory")
        if selected_trajectory:
            selected_val, optimal_val = check_selected_trajectory(
                results_dirs, selected_trajectory, task
            )
            if selected_val is not None and optimal_val is not None:
                print(
                    f"task: {task}, selected_val: {selected_val}, optimal_val: {optimal_val}"
                )
                judge_score += selected_val
                optimal_score += optimal_val
    return judge_score, optimal_score
