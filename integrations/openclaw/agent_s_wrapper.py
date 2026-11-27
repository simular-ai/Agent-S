#!/usr/bin/env python3
"""
Agent-S Wrapper for OpenClaw Integration

This script provides a simple interface for OpenClaw to invoke Agent-S
for GUI automation tasks.
"""

import argparse
import json
import subprocess
import sys
import os
import shutil
import threading


def run_agent_s(task, max_steps=15, enable_reflection=True, enable_local_env=False):
    """
    Execute an Agent-S task and return the result.

    Args:
        task: Natural language task description
        max_steps: Maximum number of steps (default: 15)
        enable_reflection: Enable reflection agent (default: True)
        enable_local_env: Enable local code execution (default: False, WARNING: executes arbitrary code)

    Returns:
        Dictionary with status and message
    """

    # Path to agent_s executable - auto-detect or use environment variable
    agent_s_path = os.environ.get("AGENT_S_PATH") or shutil.which("agent_s")
    if not agent_s_path:
        return {
            "status": "error",
            "message": "agent_s not found in PATH. Install with: pip install gui-agents",
            "error": "agent_s executable not found"
        }

    # Build base command
    cmd = [
        agent_s_path,
        "--provider", "anthropic",
        "--model", "claude-sonnet-4-5",
        "--model_temperature", "1.0",
        "--max_trajectory_length", str(max_steps),
        "--task", task,
    ]
    
    # Add optional grounding configuration from environment variables
    ground_url = os.environ.get("AGENT_S_GROUND_URL")
    ground_api_key = os.environ.get("AGENT_S_GROUND_API_KEY")
    ground_model = os.environ.get("AGENT_S_GROUND_MODEL", "ui-tars-1.5-7b")
    grounding_width = os.environ.get("AGENT_S_GROUNDING_WIDTH", "1920")
    grounding_height = os.environ.get("AGENT_S_GROUNDING_HEIGHT", "1080")
    
    if ground_url:
        cmd.extend(["--ground_provider", "huggingface"])
        cmd.extend(["--ground_url", ground_url])
        cmd.extend(["--ground_model", ground_model])
        cmd.extend(["--grounding_width", grounding_width])
        cmd.extend(["--grounding_height", grounding_height])
        if ground_api_key:
            cmd.extend(["--ground_api_key", ground_api_key])

    if enable_reflection:
        cmd.append("--enable_reflection")

    if enable_local_env:
        cmd.append("--enable_local_env")

    # --- Helper: sanitize sensitive args for safe logging ---
    def _sanitize_cmd(cmd_list):
        sanitized = []
        i = 0
        while i < len(cmd_list):
            if cmd_list[i] == "--ground_api_key" and i + 1 < len(cmd_list):
                sanitized.extend([cmd_list[i], "******"])
                i += 2
            else:
                sanitized.append(cmd_list[i])
                i += 1
        return sanitized

    # --- Determine logs directory (env override > cwd based) ---
    logs_dir = os.environ.get(
        "AGENT_S_LOGS_DIR",
        os.path.join(os.getcwd(), "logs")
    )

    print(f"Starting Agent-S with task: {task}", file=sys.stderr)
    print(f"Command: {' '.join(_sanitize_cmd(cmd))}", file=sys.stderr)

    # --- Hybrid: stream output to terminal in real time AND capture it ---
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr into stdout for ordered output
            text=True,
        )

        output_lines = []

        def _stream_and_capture():
            """Read lines from proc.stdout, print live, and accumulate."""
            for line in iter(proc.stdout.readline, ""):
                print(line, end="", flush=True)
                output_lines.append(line)

        reader = threading.Thread(target=_stream_and_capture, daemon=True)
        reader.start()
        reader.join(timeout=600)  # 10 minute timeout

        if reader.is_alive():
            proc.kill()
            reader.join()
            raise subprocess.TimeoutExpired(cmd, 600)

        proc.wait()
        captured_output = "".join(output_lines)

        if proc.returncode == 0:
            return {
                "status": "success",
                "message": f"Agent-S completed the task: {task}",
                "output": captured_output,
                "logs_directory": logs_dir,
            }
        else:
            return {
                "status": "error",
                "message": f"Agent-S failed with return code {proc.returncode}",
                "output": captured_output,
                "logs_directory": logs_dir,
            }

    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": f"Agent-S timed out after 10 minutes for task: {task}",
            "error": "Timeout expired",
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to execute Agent-S: {str(e)}",
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="OpenClaw wrapper for Agent-S GUI automation"
    )
    parser.add_argument(
        "task",
        type=str,
        help="Natural language description of the GUI task to perform"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15,
        help="Maximum number of agent steps (default: 15)"
    )
    parser.add_argument(
        "--enable-reflection",
        action="store_true",
        default=True,
        help="Enable reflection agent for better performance"
    )
    parser.add_argument(
        "--no-reflection",
        action="store_false",
        dest="enable_reflection",
        help="Disable reflection agent"
    )
    parser.add_argument(
        "--enable-local-env",
        action="store_true",
        default=False,
        help="Enable local code execution (WARNING: executes arbitrary code)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON"
    )

    args = parser.parse_args()

    # Execute Agent-S task
    result = run_agent_s(
        task=args.task,
        max_steps=args.max_steps,
        enable_reflection=args.enable_reflection,
        enable_local_env=args.enable_local_env
    )

    # Output result
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result["status"] == "success":
            print(f"\n✓ {result['message']}")
            if result.get("logs_directory"):
                print(f"📁 Logs: {result['logs_directory']}")
        else:
            print(f"\n✗ {result['message']}")
            if result.get("error"):
                print(f"\nError:\n{result['error']}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
