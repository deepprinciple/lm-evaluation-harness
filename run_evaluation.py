#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估任务管理器
支持并发执行多个评估任务，每个任务独立的错误处理和日志记录
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

# 模型配置
MODEL_CONFIGS = {
    "gpt-o3": {
        "model": "openai-chat-completions",
        "model_args": "model=o3"
    },
    "claude-opus-4-1": {
        "model": "anthropic-chat-completions", 
        "model_args": "model=claude-opus-4-1-20250805"
    },
    "grok-4": {
        "model": "xai-chat-completions",
        "model_args": "model=grok-4"
    },
    "gemini-2.5-flash": {
        "model": "gemini-chat-completions",
        "model_args": "model=gemini-2.5-flash"
    },
    "deepseek-chat": {
        "model": "deepseek-chat-completions",
        "model_args": "model=deepseek-chat"
    },
    "deepseek-reasoner": {
        "model": "deepseek-chat-completions",
        "model_args": "model=deepseek-reasoner"
    }
}

# 数据集配置
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
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建主日志文件
        self.main_log_path = os.path.join(log_dir, f"main_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
    def log_message(self, message: str, also_print: bool = True):
        """记录日志消息到主日志文件"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"
        
        with self.lock:
            with open(self.main_log_path, 'a', encoding='utf-8') as f:
                f.write(log_entry)
            
            if also_print:
                print(message)
    
    def build_command(self, model_name: str, dataset_name: str) -> List[str]:
        """构建lm_eval命令"""
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"不支持的模型: {model_name}. 支持的模型: {list(MODEL_CONFIGS.keys())}")
        
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"不支持的数据集: {dataset_name}. 支持的数据集: {list(DATASET_CONFIGS.keys())}")
        
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
            "--limit", "1"
        ]
        
        return command
    
    def run_single_task(self, model_name: str, dataset_name: str) -> Dict:
        """执行单个评估任务"""
        task_id = f"{model_name}_{dataset_name}"
        start_time = datetime.now()
        
        # 为每个任务创建独立的日志文件
        task_log_path = os.path.join(self.log_dir, f"task_{task_id}_{start_time.strftime('%Y%m%d_%H%M%S')}.txt")
        
        self.log_message(f"开始执行任务: {task_id}")
        
        try:
            # 构建命令
            command = self.build_command(model_name, dataset_name)
            command_str = " ".join(command)
            
            self.log_message(f"任务 {task_id} 执行命令: {command_str}")
            
            # 执行命令并捕获输出
            with open(task_log_path, 'w', encoding='utf-8') as log_file:
                log_file.write(f"任务: {task_id}\n")
                log_file.write(f"开始时间: {start_time}\n")
                log_file.write(f"命令: {command_str}\n")
                log_file.write("=" * 80 + "\n\n")
                log_file.flush()
                
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # 实时写入日志
                for line in process.stdout:
                    log_file.write(line)
                    log_file.flush()
                
                return_code = process.wait()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if return_code == 0:
                self.log_message(f"任务 {task_id} 成功完成 (耗时: {duration:.2f}秒)")
                status = "成功"
            else:
                self.log_message(f"任务 {task_id} 执行失败，退出码: {return_code} (耗时: {duration:.2f}秒)")
                status = "失败"
            
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
            error_msg = f"任务 {task_id} 发生异常: {str(e)} (耗时: {duration:.2f}秒)"
            self.log_message(error_msg)
            
            # 将错误信息也写入任务日志
            with open(task_log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"\n\n错误信息: {str(e)}\n")
            
            return {
                "task_id": task_id,
                "model": model_name,
                "dataset": dataset_name,
                "status": "异常",
                "error": str(e),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": duration,
                "log_file": task_log_path,
                "command": "N/A"
            }
    
    def run_tasks(self, tasks: List[Tuple[str, str]]) -> List[Dict]:
        """并发执行多个任务"""
        self.log_message(f"开始执行 {len(tasks)} 个任务，最大并发数: {self.max_workers}")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(self.run_single_task, model, dataset): (model, dataset)
                for model, dataset in tasks
            }
            
            # 收集结果
            for future in as_completed(future_to_task):
                model, dataset = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.log_message(f"任务 {model}_{dataset} 意外失败: {str(e)}")
                    results.append({
                        "task_id": f"{model}_{dataset}",
                        "model": model,
                        "dataset": dataset,
                        "status": "意外失败",
                        "error": str(e),
                        "log_file": "N/A"
                    })
        
        return results
    
    def save_summary(self, results: List[Dict]):
        """保存任务执行总结"""
        summary_path = os.path.join(self.log_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        summary = {
            "总任务数": len(results),
            "成功任务数": len([r for r in results if r["status"] == "成功"]),
            "失败任务数": len([r for r in results if r["status"] in ["失败", "异常", "意外失败"]]),
            "执行时间": datetime.now().isoformat(),
            "详细结果": results
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        self.log_message(f"任务执行总结已保存到: {summary_path}")
        return summary_path

def main():
    parser = argparse.ArgumentParser(description="模型评估任务管理器")
    parser.add_argument("--models", nargs="+", required=True,
                       choices=list(MODEL_CONFIGS.keys()),
                       help="要评估的模型列表")
    parser.add_argument("--datasets", nargs="+", required=True,
                       choices=list(DATASET_CONFIGS.keys()),
                       help="要使用的数据集列表")
    parser.add_argument("--max-workers", type=int, default=2,
                       help="最大并发任务数 (默认: 2)")
    parser.add_argument("--log-dir", default="logs",
                       help="日志文件保存目录 (默认: logs)")
    
    args = parser.parse_args()
    
    # 创建任务管理器
    task_manager = TaskManager(max_workers=args.max_workers, log_dir=args.log_dir)
    
    # 生成任务列表（模型和数据集的笛卡尔积）
    tasks = [(model, dataset) for model in args.models for dataset in args.datasets]
    
    task_manager.log_message(f"准备执行的任务组合:")
    for model, dataset in tasks:
        task_manager.log_message(f"  - {model} x {dataset}")
    
    # 执行任务
    results = task_manager.run_tasks(tasks)
    
    # 保存总结
    summary_path = task_manager.save_summary(results)
    
    # 打印最终统计
    successful = [r for r in results if r["status"] == "成功"]
    failed = [r for r in results if r["status"] in ["失败", "异常", "意外失败"]]
    
    print(f"\n{'='*60}")
    print(f"任务执行完成!")
    print(f"总任务数: {len(results)}")
    print(f"成功: {len(successful)}")
    print(f"失败: {len(failed)}")
    print(f"主日志文件: {task_manager.main_log_path}")
    print(f"执行总结: {summary_path}")
    print(f"{'='*60}")
    
    # 如果有失败的任务，以非零状态退出
    if failed:
        sys.exit(1)

if __name__ == "__main__":
    main()
