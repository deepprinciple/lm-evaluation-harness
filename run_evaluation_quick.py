#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Evaluation Task Manager - Quick Test Version (--limit 1)
Supports concurrent execution of multiple evaluation tasks, each with independent error handling and logging
"""

import argparse
import subprocess
import json
import os
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import threading

# Model configurations
MODEL_CONFIGS = {
    "gpt-o3": {
        "model": "openai-chat-completions",
        "model_args": "model=o3"
    },
    "gpt-5": {
        "model": "openai-chat-completions",
        "model_args": "model=gpt-5"
    },
    "claude-opus-4-1": {
        "model": "anthropic-chat-completions", 
        "model_args": "model=claude-opus-4-1-20250805"
    },
    "grok-4": {
        "model": "xai-chat-completions",
        "model_args": "model=grok-4"
    },
    "grok-3-mini": {
        "model": "xai-chat-completions",
        "model_args": "model=grok-3-mini"
    },
    "grok-3": {
        "model": "xai-chat-completions",
        "model_args": "model=grok-3"
    },
    "gemini-2.5-flash": {
        "model": "gemini-chat-completions",
        "model_args": "model=gemini-2.5-flash"
    },
    "gemini-2.5-pro": {
        "model": "gemini-chat-completions",
        "model_args": "model=gemini-2.5-pro"
    },
    "deepseek-chat": {
        "model": "deepseek-chat-completions",
        "model_args": "model=deepseek-chat"
    },
    "deepseek-reasoner": {
        "model": "deepseek-chat-completions",
        "model_args": "model=deepseek-reasoner"
    },
    "claude-4-5-sonnet": {
        "model": "anthropic-chat-completions",
        "model_args": "model=claude-sonnet-4-5-20250929"
    },
}

# Dataset configurations
DATASET_CONFIGS = {
    "chemistry": {
        "task": "science_chemistry",
        "output_dir": "test/result/Chemistry"
    },
    "biology": {
        "task": "science_biology", 
        "output_dir": "test/result/Biology"
    },
    "materials": {
        "task": "science_materials",
        "output_dir": "test/result/Material"
    }
}

class TaskManager:
    def __init__(self, max_workers=2, log_dir="logs"):
        self.max_workers = max_workers
        self.log_dir = log_dir
        self.lock = threading.Lock()
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Create main log file
        self.main_log_path = os.path.join(log_dir, f"main_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
    def log_message(self, message: str, also_print: bool = True):
        """Log message to main log file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"
        
        with self.lock:
            with open(self.main_log_path, 'a', encoding='utf-8') as f:
                f.write(log_entry)
            
            if also_print:
                print(message)
    
    def build_command(self, model_name: str, dataset_name: str) -> List[str]:
        """Build lm_eval command - Quick version (--limit 1)"""
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(MODEL_CONFIGS.keys())}")

        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {list(DATASET_CONFIGS.keys())}")
        
        model_config = MODEL_CONFIGS[model_name]
        dataset_config = DATASET_CONFIGS[dataset_name]
        
        command = [
            "lm_eval",
            "--model", model_config["model"],
            "--model_args", model_config["model_args"],
            "--include_path", "lm_eval/tasks",
            "--tasks", dataset_config["task"],
            "--apply_chat_template",
            "--output", dataset_config["output_dir"],
            "--log_samples",
            "--limit", "1",  # Quick test version: evaluate only 1 sample
        ]
        
        return command
    
    def run_single_task(self, model_name: str, dataset_name: str) -> Dict:
        """Execute a single evaluation task"""
        task_id = f"{model_name}_{dataset_name}"
        start_time = datetime.now()

        # Create independent log file for each task
        task_log_path = os.path.join(self.log_dir, f"task_{task_id}_{start_time.strftime('%Y%m%d_%H%M%S')}.txt")

        self.log_message(f"Starting task: {task_id} (Quick mode: limit 1 sample)")
        
        try:
            # Build command
            command = self.build_command(model_name, dataset_name)
            command_str = " ".join(command)

            self.log_message(f"Task {task_id} executing command: {command_str}")

            # Execute command and capture output
            with open(task_log_path, 'w', encoding='utf-8') as log_file:
                log_file.write(f"Task: {task_id}\n")
                log_file.write(f"Start time: {start_time}\n")
                log_file.write(f"Command: {command_str}\n")
                log_file.write("=" * 80 + "\n\n")
                log_file.flush()
                
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )

                # Write logs in real-time
                for line in process.stdout:
                    log_file.write(line)
                    log_file.flush()
                
                return_code = process.wait()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if return_code == 0:
                self.log_message(f"Task {task_id} completed successfully (elapsed time: {duration:.2f}s)")
                status = "Success"
            else:
                self.log_message(f"Task {task_id} failed, exit code: {return_code} (elapsed time: {duration:.2f}s)")
                status = "Failed"
            
            return {
                "task_id": task_id,
                "model": model_name,
                "dataset": dataset_name,
                "status": status,
                "return_code": return_code,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": duration,
                "log_file": task_log_path,
                "command": command_str
            }
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            error_msg = f"Task {task_id} encountered exception: {str(e)} (elapsed time: {duration:.2f}s)"
            self.log_message(error_msg)

            # Write error information to task log as well
            with open(task_log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"\n\nError information: {str(e)}\n")
            
            return {
                "task_id": task_id,
                "model": model_name,
                "dataset": dataset_name,
                "status": "Exception",
                "error": str(e),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": duration,
                "log_file": task_log_path,
                "command": "N/A"
            }
    
    def run_tasks(self, tasks: List[Tuple[str, str]]) -> List[Dict]:
        """Execute multiple tasks concurrently"""
        self.log_message(f"Starting execution of {len(tasks)} tasks, max concurrency: {self.max_workers} (Quick mode)")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.run_single_task, model, dataset): (model, dataset)
                for model, dataset in tasks
            }

            # Collect results
            for future in as_completed(future_to_task):
                model, dataset = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.log_message(f"Task {model}_{dataset} unexpected failure: {str(e)}")
                    results.append({
                        "task_id": f"{model}_{dataset}",
                        "model": model,
                        "dataset": dataset,
                        "status": "Unexpected failure",
                        "error": str(e),
                        "log_file": "N/A"
                    })
        
        return results
    
    def save_summary(self, results: List[Dict]):
        """Save task execution summary"""
        summary_path = os.path.join(self.log_dir, f"summary_quick_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        summary = {
            "Mode": "Quick test mode (--limit 1)",
            "Total tasks": len(results),
            "Successful tasks": len([r for r in results if r["status"] == "Success"]),
            "Failed tasks": len([r for r in results if r["status"] in ["Failed", "Exception", "Unexpected failure"]]),
            "Execution time": datetime.now().isoformat(),
            "Detailed results": results
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        self.log_message(f"Task execution summary saved to: {summary_path}")
        return summary_path

def main():
    parser = argparse.ArgumentParser(description="Model Evaluation Task Manager - Quick Test Version")
    parser.add_argument("--models", nargs="+",
                       choices=list(MODEL_CONFIGS.keys()),
                       help="List of models to evaluate")
    parser.add_argument("--datasets", nargs="+",
                       choices=list(DATASET_CONFIGS.keys()),
                       help="List of datasets to use")
    parser.add_argument("--all", action="store_true",
                       help="Test all combinations of models and datasets")
    parser.add_argument("--max-workers", type=int, default=2,
                       help="Maximum number of concurrent tasks (default: 2)")
    parser.add_argument("--log-dir", default="logs",
                       help="Directory to save log files (default: logs)")
    
    args = parser.parse_args()

    # Parameter validation
    if args.all:
        # If --all is specified, use all models and datasets
        models = list(MODEL_CONFIGS.keys())
        datasets = list(DATASET_CONFIGS.keys())
    else:
        # Otherwise check if model and dataset parameters are provided
        if not args.models or not args.datasets:
            parser.error("Must specify --models and --datasets, or use --all parameter")
        models = args.models
        datasets = args.datasets

    # Create task manager
    task_manager = TaskManager(max_workers=args.max_workers, log_dir=args.log_dir)

    # Generate task list (Cartesian product of models and datasets)
    tasks = [(model, dataset) for model in models for dataset in datasets]
    
    if args.all:
        task_manager.log_message(f"Task combinations to execute (Quick mode: --limit 1) - Using all models and datasets:")
        task_manager.log_message(f"Models: {', '.join(models)}")
        task_manager.log_message(f"Datasets: {', '.join(datasets)}")
        task_manager.log_message(f"Total {len(tasks)} task combinations:")
    else:
        task_manager.log_message(f"Task combinations to execute (Quick mode: --limit 1):")
    
    for model, dataset in tasks:
        task_manager.log_message(f"  - {model} x {dataset}")

    # Execute tasks
    results = task_manager.run_tasks(tasks)

    # Save summary
    summary_path = task_manager.save_summary(results)

    # Print final statistics
    successful = [r for r in results if r["status"] == "Success"]
    failed = [r for r in results if r["status"] in ["Failed", "Exception", "Unexpected failure"]]
    
    print(f"\n{'='*60}")
    print(f"Quick test mode task execution completed! (--limit 1)")
    print(f"Total tasks: {len(results)}")
    print(f"Success: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Main log file: {task_manager.main_log_path}")
    print(f"Execution summary: {summary_path}")
    print(f"{'='*60}")

    # Exit with non-zero status if there are failed tasks
    if failed:
        sys.exit(1)

if __name__ == "__main__":
    main()
