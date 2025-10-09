#!/bin/bash

# 模型评估任务执行脚本
# 使用方法: ./run_evaluation.sh [选项]

set -e  # 遇到错误时退出

# 设置API密钥
# 请在环境变量中设置以下API密钥，或在此处取消注释并填入您的密钥
# export OPENAI_API_KEY=your_openai_api_key_here
# export ANTHROPIC_API_KEY=your_anthropic_api_key_here
# export DEEPSEEK_API_KEY=your_deepseek_api_key_here
# export XAI_API_KEY=your_xai_api_key_here
# export GEMINI_API_KEY=your_gemini_api_key_here
# export QWEN_API_KEY=your_qwen_api_key_here

# 脚本目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON_SCRIPT="$SCRIPT_DIR/run_evaluation.py"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
模型评估任务执行脚本

使用方法:
  $0 [选项]

选项:
  -m, --models MODEL1,MODEL2,...    要评估的模型列表 (用逗号分隔)
  -d, --datasets DATASET1,DATASET2,...  要使用的数据集列表 (用逗号分隔)
  -w, --max-workers NUMBER          最大并发任务数 (默认: 2)
  -l, --log-dir DIRECTORY          日志文件保存目录 (默认: logs)
  -h, --help                       显示此帮助信息

支持的模型:
  - gpt-o3                        GPT O3
  - claude-opus-4-1               Claude Opus 4.1
  - grok-4                        Grok 4
  - gemini-2.5-flash              Gemini 2.5 Flash
  - deepseek-chat                 DeepSeek Chat
  - deepseek-reasoner             DeepSeek Reasoner

支持的数据集:
  - chemistry                     化学数据集
  - biology                       生物数据集
  - materials                     材料数据集

示例:
  # 使用GPT-O3和Claude在所有数据集上评估
  $0 -m gpt-o3,claude-opus-4-1 -d chemistry,biology,materials

  # 使用所有模型在化学数据集上评估，并发数为3
  $0 -m gpt-o3,claude-opus-4-1,grok-4,gemini-2.5-flash,deepseek-chat,deepseek-reasoner -d chemistry -w 3

  # 快速测试：一个模型一个数据集
  $0 -m gpt-o3 -d chemistry

预设配置:
  all-models     所有支持的模型
  all-datasets   所有支持的数据集
  
  使用示例:
  $0 -m all-models -d chemistry
  $0 -m gpt-o3 -d all-datasets

EOF
}

# 展开预设配置
expand_presets() {
    local input="$1"
    if [[ "$input" == "all-models" ]]; then
        echo "gpt-o3,claude-opus-4-1,grok-4,gemini-2.5-flash,deepseek-chat,deepseek-reasoner"
    elif [[ "$input" == "all-datasets" ]]; then
        echo "chemistry,biology,materials"
    else
        echo "$input"
    fi
}

# 验证输入参数
validate_models() {
    local models="$1"
    local valid_models="gpt-o3 claude-opus-4-1 grok-4 gemini-2.5-flash deepseek-chat deepseek-reasoner"
    
    IFS=',' read -ra MODEL_ARRAY <<< "$models"
    for model in "${MODEL_ARRAY[@]}"; do
        if [[ ! " $valid_models " =~ " $model " ]]; then
            print_error "无效的模型: $model"
            print_info "支持的模型: $valid_models"
            return 1
        fi
    done
    return 0
}

validate_datasets() {
    local datasets="$1"
    local valid_datasets="chemistry biology materials"
    
    IFS=',' read -ra DATASET_ARRAY <<< "$datasets"
    for dataset in "${DATASET_ARRAY[@]}"; do
        if [[ ! " $valid_datasets " =~ " $dataset " ]]; then
            print_error "无效的数据集: $dataset"
            print_info "支持的数据集: $valid_datasets"
            return 1
        fi
    done
    return 0
}

# 检查Python脚本是否存在
check_python_script() {
    if [[ ! -f "$PYTHON_SCRIPT" ]]; then
        print_error "Python脚本不存在: $PYTHON_SCRIPT"
        return 1
    fi
    
    if [[ ! -x "$PYTHON_SCRIPT" ]]; then
        print_info "设置Python脚本为可执行..."
        chmod +x "$PYTHON_SCRIPT"
    fi
    
    return 0
}

# 主函数
main() {
    local models=""
    local datasets=""
    local max_workers=2
    local log_dir="logs"
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--models)
                models="$2"
                shift 2
                ;;
            -d|--datasets)
                datasets="$2"
                shift 2
                ;;
            -w|--max-workers)
                max_workers="$2"
                shift 2
                ;;
            -l|--log-dir)
                log_dir="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                print_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 检查必需参数
    if [[ -z "$models" ]]; then
        print_error "请指定要评估的模型 (-m/--models)"
        show_help
        exit 1
    fi
    
    if [[ -z "$datasets" ]]; then
        print_error "请指定要使用的数据集 (-d/--datasets)"
        show_help
        exit 1
    fi
    
    # 展开预设配置
    models=$(expand_presets "$models")
    datasets=$(expand_presets "$datasets")
    
    print_info "配置检查..."
    print_info "模型: $models"
    print_info "数据集: $datasets"
    print_info "最大并发数: $max_workers"
    print_info "日志目录: $log_dir"
    
    # 验证参数
    if ! validate_models "$models"; then
        exit 1
    fi
    
    if ! validate_datasets "$datasets"; then
        exit 1
    fi
    
    # 检查Python脚本
    if ! check_python_script; then
        exit 1
    fi
    
    # 创建日志目录
    mkdir -p "$log_dir"
    
    # 转换为Python脚本参数格式
    IFS=',' read -ra MODEL_ARRAY <<< "$models"
    IFS=',' read -ra DATASET_ARRAY <<< "$datasets"
    
    # 构建Python命令
    python_args=()
    python_args+=("--models")
    python_args+=("${MODEL_ARRAY[@]}")
    python_args+=("--datasets") 
    python_args+=("${DATASET_ARRAY[@]}")
    python_args+=("--max-workers" "$max_workers")
    python_args+=("--log-dir" "$log_dir")
    
    print_info "开始执行评估任务..."
    print_info "执行命令: python3 $PYTHON_SCRIPT ${python_args[*]}"
    
    # 执行Python脚本
    if python3 "$PYTHON_SCRIPT" "${python_args[@]}"; then
        print_success "所有任务执行完成!"
    else
        exit_code=$?
        print_warning "部分任务执行失败 (退出码: $exit_code)"
        print_info "请查看日志文件了解详细信息: $log_dir/"
        exit $exit_code
    fi
}

# 执行主函数
main "$@"
