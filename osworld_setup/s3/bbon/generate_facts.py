import os
import json
import asyncio
import argparse
from typing import List, Optional
from dotenv import load_dotenv

from gui_agents.s3.bbon.behavior_narrator import BehaviorNarrator
from utils import get_new_tasks_classification

load_dotenv()


async def generate_single_fact_caption(
    task_dir: str,
    screenshot_files: List[str],
    i: int,
    judge: BehaviorNarrator,
    trajectory_lines: List[str],
):
    """Generate a single fact caption for a screenshot pair."""
    before_file = os.path.join(task_dir, screenshot_files[i])
    after_file = os.path.join(task_dir, screenshot_files[i + 1])

    # Load action from trajectory data if available
    pyautogui_action = None
    if i < len(trajectory_lines):
        try:
            data = json.loads(trajectory_lines[i])
            pyautogui_action = data.get("exec_code")
        except:
            pass

    if pyautogui_action is None:
        raise ValueError(f"No pyautogui action found for step {i+1}")

    # Read image bytes
    try:
        with open(before_file, "rb") as f:
            before_bytes = f.read()
        with open(after_file, "rb") as f:
            after_bytes = f.read()
    except Exception as e:
        raise Exception(f"Error reading images: {e}")

    # Generate fact caption using behavior narrator
    result = await asyncio.to_thread(
        judge.judge,
        screenshot_num=i + 1,
        before_img_bytes=before_bytes,
        after_img_bytes=after_bytes,
        pyautogui_action=pyautogui_action,
    )
    result["screenshot_num"] = i + 1

    return result


async def generate_fact_captions_parallel(
    task_dir: str,
    judge: BehaviorNarrator,
    step_semaphore: Optional[asyncio.Semaphore] = None,
):
    """Generate fact captions for a task directory when they don't exist (parallelized version)."""
    print(f"Generating fact captions for {task_dir}...")

    # Find all screenshot files
    screenshot_files = []
    for filename in os.listdir(task_dir):
        if filename.startswith("step_") and filename.endswith(".png"):
            screenshot_files.append(filename)

    # Sort by step number
    def extract_step_num(filename):
        try:
            return int(filename.split("_")[1].split(".")[0])
        except:
            return 0

    screenshot_files.sort(key=extract_step_num)

    if len(screenshot_files) < 2:
        print(f"Not enough screenshots to generate fact captions in {task_dir}")
        return []

    # Load trajectory data once
    trajectory_lines = []
    trajectory_file = os.path.join(task_dir, "traj.jsonl")
    if os.path.exists(trajectory_file):
        try:
            with open(trajectory_file, "r") as f:
                trajectory_lines = f.readlines()
        except:
            pass

    # Use shared semaphore to limit concurrent judge calls
    if step_semaphore is None:
        step_semaphore = asyncio.Semaphore(5)  # Default limit

    async def bounded_task(task_func, *args, **kwargs):
        async with step_semaphore:
            return await task_func(*args, **kwargs)

    try:
        # Create bounded tasks for parallel execution
        bounded_tasks = [
            bounded_task(
                generate_single_fact_caption,
                task_dir,
                screenshot_files,
                i,
                judge,
                trajectory_lines,
            )
            for i in range(len(screenshot_files) - 1)
        ]
        results = await asyncio.gather(*bounded_tasks, return_exceptions=True)
    except Exception as e:
        print(f"Error in parallel execution: {e}")
        return []

    # Process results and save to file
    fact_captions = []
    successful_results = []
    fact_captions_file = os.path.join(task_dir, "fact_captions.jsonl")

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Error generating fact caption for step {i+1}: {result}")
            continue
        successful_results.append(result)
        fact_caption = f"Fact Caption from Screenshot {result['screenshot_num']}: {result['fact_answer']}"
        fact_captions.append(fact_caption)

    # Save all results to file at once
    if successful_results:
        with open(fact_captions_file, "w") as f:
            for result in successful_results:
                f.write(json.dumps(result) + "\n")

    print(f"Generated {len(fact_captions)} fact captions for {task_dir}")
    return fact_captions


async def main(engine_params: dict, results_dirs: List[str]):
    """Main function to generate fact captions for multiple task directories.

    Args:
        engine_params: Engine parameters for BehaviorNarrator
        results_dirs: List of results directories to analyze for task classification
    """
    # Get task IDs automatically using get_new_tasks_classification
    tasks_classification = get_new_tasks_classification(results_dirs)
    task_ids = tasks_classification["variance"]

    print(f"Found {len(task_ids)} variance tasks to process")
    judge = BehaviorNarrator(engine_params=engine_params)

    # Get concurrency settings from environment
    per_step = int(os.getenv("DIFFCAP_PER_STEP_CONCURRENCY", "100"))
    per_taskdir = int(os.getenv("DIFFCAP_PER_TASKDIR_CONCURRENCY", "4"))

    # Build list of task directories to process
    task_dirs = []
    for task_id in task_ids:
        domain, example_id = task_id.split("/")

        # Check each results directory for this task
        for results_dir in results_dirs:
            task_dir = os.path.join(results_dir, domain, example_id)

            try:
                if "fact_captions.jsonl" in os.listdir(task_dir):
                    print(f"Fact captions already exist for {task_dir}")
                    continue
            except FileNotFoundError:
                continue

            task_dirs.append(task_dir)

    if not task_dirs:
        print("No new task directories to process.")
        return

    print(f"Scheduling {len(task_dirs)} task directories...")

    # Set up semaphores for concurrency control
    shared_step_semaphore = asyncio.Semaphore(per_step)
    taskdir_semaphore = asyncio.Semaphore(per_taskdir)

    async def run_one(task_dir):
        async with taskdir_semaphore:
            print(f"Processing {task_dir}")
            return await generate_fact_captions_parallel(
                task_dir, judge, step_semaphore=shared_step_semaphore
            )

    # Execute all tasks in parallel
    results = await asyncio.gather(
        *[run_one(d) for d in task_dirs], return_exceptions=True
    )

    # Report results
    failures = sum(1 for r in results if isinstance(r, Exception))
    if failures:
        print(
            f"Completed with {failures} failures out of {len(task_dirs)} task directories."
        )
    else:
        print("Completed all task directories successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate fact captions for OSWorld task directories"
    )
    parser.add_argument(
        "--results-dirs",
        nargs="+",
        required=True,
        help="List of results directories to analyze for task classification",
    )
    parser.add_argument(
        "--model", default="gpt-5-2025-08-07", help="Model to use for generation"
    )
    parser.add_argument("--engine-type", default="openai", help="Engine type")
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for generation"
    )

    args = parser.parse_args()

    # Engine parameters
    engine_params = {
        "model": args.model,
        "engine_type": args.engine_type,
        "temperature": args.temperature,
    }

    print(f"Results directories: {args.results_dirs}")
    asyncio.run(main(engine_params, args.results_dirs))
