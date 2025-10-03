import json
import os
import asyncio
import argparse
import concurrent.futures
from typing import List, Tuple, Optional
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

from utils import (
    get_new_tasks_classification,
    evaluate_comparative_results,
    load_task_instruction,
    load_facts,
)
from gui_agents.s3.bbon.comparative_judge import ComparativeJudge


def run_judge(
    task: str, task_instruction: str, result_dirs: List[str], judge: ComparativeJudge
) -> Tuple[str, str, Optional[str]]:
    """
    Fact captions + initial/final screenshots judging.
    Pipeline: load trajectories → load existing fact captions → include initial/final screenshots → judge.
    """
    # 1. Use provided task instruction
    # task_instruction is now a direct input parameter

    # 2. Load fact captions for all trajectories
    all_fact_captions = []
    for result_dir in result_dirs:
        task_dir = os.path.join(result_dir, task.split("/")[0], task.split("/")[1])
        fact_captions = load_facts(task_dir)
        all_fact_captions.append(fact_captions)

    # 3. Use the new Judge class method
    return judge.judge(task_instruction, task, result_dirs, all_fact_captions)


def evaluate_trajectories(
    task: str, task_instruction: str, result_dirs: List[str], judge: ComparativeJudge
) -> Tuple[str, str, dict]:
    """Wrapper that runs fact-only MCQ judge and returns results."""
    answer, thoughts, selected_trajectory = run_judge(
        task, task_instruction, result_dirs, judge
    )

    record = {
        "selected_trajectory": selected_trajectory,
        "answer": answer,
        "thoughts": thoughts,
    }

    print(f"✅ Added task {task} (MCQ fact-only)")
    return answer, thoughts, record


asyncio.get_event_loop().set_default_executor(
    concurrent.futures.ThreadPoolExecutor(max_workers=100)
)


async def run_async(
    task: str, task_instruction: str, result_dirs: List[str], judge: ComparativeJudge
):
    """Async wrapper for fact-only MCQ evaluation."""
    return await asyncio.to_thread(
        evaluate_trajectories,
        task=task,
        task_instruction=task_instruction,
        result_dirs=result_dirs,
        judge=judge,
    )


async def evaluate_and_save(
    result_dirs: List[str],
    output_file_path: str,
    examples_path: str,
    engine_params: dict,
):
    """Main evaluation function that processes tasks and saves results."""
    res = get_new_tasks_classification(results_dirs=result_dirs)
    for key in res:
        print(f"{key}: {res[key]}")
    optimal, minimum, expected_value = (
        res["optimal"],
        res["minimum"],
        res["expected_value"],
    )
    print(f"optimal score: {optimal}, minimum score: {minimum}")

    variance = res["variance"]

    judge = ComparativeJudge(engine_params=engine_params)

    # Load existing results
    if os.path.exists(output_file_path):
        with open(output_file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, dict):
                    data = {}
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    # Prepare async tasks only for tasks not yet in data
    tasks = []
    task_names = []
    for task in variance:
        if str(task) in data:
            print(f"⚠️ Task {task} already exists in results — skipping.")
            continue

        # Load task instruction from examples path
        task_instruction = load_task_instruction(task, examples_path)
        if task_instruction is None:
            print(f"⚠️ No task instruction found for {task}, skipping...")
            continue

        tasks.append(run_async(task, task_instruction, result_dirs, judge))
        task_names.append(task)

    # Run only new tasks
    results = await tqdm_asyncio.gather(*tasks)
    # Merge into existing results
    for task, (ans, thoughts, record) in zip(task_names, results):
        data[str(task)] = record

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as f:
        json.dump(data, f, indent=2)

    res = evaluate_comparative_results(result_dirs, json_path=output_file_path)
    gain, maximum_gain = res
    data["score"] = {
        "optimal": optimal,
        "minimum": minimum,
        "expected_value": expected_value,
        "res": res,
        "actual score": minimum + gain,
    }
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as f:
        json.dump(data, f, indent=2)

    return results


async def run_experiment(
    shuffled_runs: List[str],
    output_dir: str,
    examples_path: str,
    engine_params: dict,
    start_round: int = 2,
    max_rounds: int = None,
):
    """
    Run fact-only experiments progressively: start_round vs start_round+1, etc.
    """
    if max_rounds is None:
        max_rounds = len(shuffled_runs)

    os.makedirs(output_dir, exist_ok=True)

    for i in range(start_round, max_rounds + 1):  # start at start_round (default 2)
        test_dirs = shuffled_runs[:i]
        output_file_path = os.path.join(output_dir, f"BoN{i}.json")

        print(f"Running fact-only experiment with {i} dirs → {output_file_path}")
        await evaluate_and_save(
            test_dirs, output_file_path, examples_path, engine_params
        )


async def main(
    shuffled_runs: List[str] = None,
    output_dir: str = None,
    examples_path: str = None,
    engine_params: dict = None,
    start_round: int = 2,
    max_rounds: int = None,
):
    """Main function to run fact-only judge experiments.

    Args:
        shuffled_runs: List of result directory paths to compare
        output_dir: Directory to save results
        examples_path: Path to examples directory containing task instructions
        engine_params: Engine parameters for the judge
        start_round: Starting round number (default: 2)
        max_rounds: Maximum number of rounds to run (default: len(shuffled_runs))
    """
    if shuffled_runs is None:
        print("Error: shuffled_runs must be provided")
        return

    if output_dir is None:
        print("Error: output_dir must be provided")
        return

    if examples_path is None:
        print("Error: examples_path must be provided")
        return

    if engine_params is None:
        print("Error: engine_params must be provided")
        return

    await run_experiment(
        shuffled_runs, output_dir, examples_path, engine_params, start_round, max_rounds
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run fact-only judge experiments on OSWorld task directories"
    )
    parser.add_argument(
        "--results-dirs",
        nargs="+",
        required=True,
        help="List of results directories to analyze",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to save results")
    parser.add_argument(
        "--examples-path",
        required=True,
        help="Path to examples directory containing task instructions",
    )
    parser.add_argument(
        "--start-round", type=int, default=2, help="Starting round number (default: 2)"
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Maximum number of rounds to run (default: len(results_dirs))",
    )
    parser.add_argument(
        "--model", default="gpt-5-2025-08-07", help="Model to use for judging"
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
    print(f"Output directory: {args.output_dir}")
    print(f"Examples path: {args.examples_path}")
    print(f"Start round: {args.start_round}")
    print(f"Max rounds: {args.max_rounds}")
    print(f"Engine params: {engine_params}")

    # Run fact-only evaluation
    asyncio.run(
        main(
            shuffled_runs=args.results_dirs,
            output_dir=args.output_dir,
            examples_path=args.examples_path,
            engine_params=engine_params,
            start_round=args.start_round,
            max_rounds=args.max_rounds,
        )
    )
