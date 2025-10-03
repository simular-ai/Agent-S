# Deplying Agent S3 in OSWorld

# Step 1: Set up Agent S3

Follow the [README.md](https://github.com/simular-ai/Agent-S/blob/main/README.md) to set up Agent S3.

# Step 2: Copying Over Run Files

If you haven't already, please follow the [OSWorld environment setup](https://github.com/xlang-ai/OSWorld/blob/main/README.md). We've provided the relevant OSWorld run files for evaluation in this `osworld_setup` folder. Please copy this over to your OSWorld folder. `run_local.py` is for if you want to run locally on VMWare and `run.py` and `lib_run_single.py` are for if you want to run on AWS. All run commands in order are provided in the `run.sh`. Copy over the files in `osworld_setup/s3/bbon` as well. 

# Step 3: Switch the AMI 

Switch image AMI for the AWS provider in `desktop_env/providers/aws/manager.py` is set to `"ami-0b505e9d0d99ba88c"`.

# Step 4: Generating Facts

After completing your OSWorld runs and having result directories, run `generate_facts.py` to generate fact captions for screenshot pairs:

```bash
python osworld_setup/s3/bbon/generate_facts.py \
  --results-dirs \
    results1/pyautogui/screenshot/gpt-5-2025-08-07 \
    results2/pyautogui/screenshot/gpt-5-2025-08-07 \
  --model "gpt-5-2025-08-07" \
  --engine-type "openai" \
  --temperature 1.0
```

This will populate your result directories with `fact_captions.jsonl` files containing behavioral descriptions of screenshot differences.

# Step 5: Run the Judge

Finally, run `run_judge.py` to evaluate the trajectories using the generated fact captions:

```bash
python osworld_setup/s3/bbon/run_judge.py \
  --results-dirs \
    results1/pyautogui/screenshot/gpt-5-2025-08-07 \
    results2/pyautogui/screenshot/gpt-5-2025-08-07 \
  --output-dir "judge_results" \
  --examples-path "evaluation_examples/examples" \
  --model "gpt-5-2025-08-07" \
  --engine-type "openai" \
  --temperature 1.0
```

This will:
- Compare trajectories across different result directories
- Use the facts to judge which trajectory performs better
- Generate evaluation results
- Save results to the specified output directory

The judge will create files like `BoN2.json`, `BoN3.json`, etc., showing the performance comparison as you add more trajectories.

