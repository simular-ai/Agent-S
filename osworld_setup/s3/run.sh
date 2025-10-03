# Step 1: Complete 2 or more rollouts on either AWS or locally
python run.py \
  --provider_name "aws" \
  --headless \
  --num_envs 10 \
  --max_steps 100 \
  --domain "all" \
  --test_all_meta_path evaluation_examples/test_nogdrive.json \
  --result_dir "results" \
  --region "us-east-1" \
  --model_provider "openai" \
  --model "gpt-5-2025-08-07" \
  --model_temperature 1.0 \
  --ground_provider "huggingface" \
  --ground_url "<YOUR_HUGGINGFACE_ENDPOINT_URL>/v1" \
  --grounding_width 1920 \
  --grounding_height 1080 \
  --sleep_after_execution 3

python run_local.py \
  --path_to_vm "/Users/user/OSWorld/vmware_vm_data/Ubuntu0/Ubuntu0.vmx" \
  --provider_name "vmware" \
  --headless \
  --max_steps 100 \
  --domain "all" \
  --test_all_meta_path evaluation_examples/test_nogdrive.json \
  --result_dir "results" \
  --model_provider "openai" \
  --model "gpt-5-2025-08-07" \
  --model_temperature 1.0 \
  --ground_provider "huggingface" \
  --ground_url "<YOUR_HUGGINGFACE_ENDPOINT_URL>/v1" \
  --grounding_width 1920 \
  --grounding_height 1080

# Step 2: Generate Facts
python generate_facts.py \
  --results-dirs \
    results1/pyautogui/screenshot/gpt-5-2025-08-07 \
    results2/pyautogui/screenshot/gpt-5-2025-08-07 \
  --model "gpt-5-2025-08-07" \
  --engine-type "openai" \
  --temperature 1.0

# Step 3: Run the Judge. Make sure the order of the results-dirs is the same as the order above.
python run_judge.py \
  --results-dirs \
    results1/pyautogui/screenshot/gpt-5-2025-08-07 \
    results2/pyautogui/screenshot/gpt-5-2025-08-07 \
  --output-dir "judge_results" \
  --examples-path "evaluation_examples/examples" \
  --model "gpt-5-2025-08-07" \
  --engine-type "openai" \
  --temperature 1.0