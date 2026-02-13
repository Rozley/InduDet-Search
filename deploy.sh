#!/bin/bash
# InduDet-Search 快速部署脚本
# 用法: bash deploy.sh [GPU数量]

set -e  # 遇到错误立即退出

echo "========================================"
echo "InduDet-Search 快速部署脚本"
echo "========================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认配置
NUM_GPUS=${1:-1}
PROJECT_DIR=$(pwd)
DATA_DIR="${PROJECT_DIR}/data"
RESULTS_DIR="${PROJECT_DIR}/results"

# 输出配置函数
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 检查系统
check_system() {
    log_info "检查系统环境..."

    # 检查Python
    if ! command -v python &> /dev/null; then
        log_error "Python 未安装，请先安装 Python 3.8+"
        exit 1
    fi
    PYTHON_VERSION=$(python -c 'import sys; print(sys.version_info.major)')
    log_info "Python 版本: $(python --version)"

    # 检查CUDA
    if command -v nvidia-smi &> /dev/null; then
        log_info "NVIDIA GPU 可用:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    else
        log_warn "未检测到 NVIDIA GPU，将使用 CPU 模式（速度较慢）"
    fi
}

# 安装依赖
install_dependencies() {
    log_info "安装 Python 依赖包..."

    # 创建虚拟环境（推荐）
    if [ ! -d "venv" ]; then
        log_info "创建 Python 虚拟环境..."
        python -m venv venv
    fi

    # 激活虚拟环境
    source venv/bin/activate  # Linux/Mac
    # source venv/Scripts/activate  # Windows

    # 升级pip
    pip install --upgrade pip

    # 安装依赖
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || \
    pip install torch torchvision

    pip install -r requirements.txt

    log_info "依赖安装完成!"
}

# 准备数据集
prepare_data() {
    log_info "检查数据集..."

    if [ ! -d "${DATA_DIR}" ]; then
        mkdir -p "${DATA_DIR}"
    fi

    # 检查MVTec AD数据集
    CATEGORIES=$(ls "${DATA_DIR}" 2>/dev/null | wc -l)
    if [ "${CATEGORIES}" -eq 0 ]; then
        log_warn "未检测到数据集!"
        echo ""
        echo "请选择以下方式之一准备数据:"
        echo "1. 手动下载 MVTec AD: https://www.mvtec.com/company/research/datasets/mvtec-ad"
        echo "2. 解压已下载的数据集到 ${DATA_DIR}"
        echo "3. 使用示例数据集（自动生成）"
        echo ""
        read -p "请选择 [1/2/3]: " choice
        case $choice in
            1)
                log_info "请手动下载数据集后解压到 ${DATA_DIR}"
                ;;
            2)
                log_info "请将数据集解压后重新运行此脚本"
                ;;
            3)
                log_info "创建示例数据集..."
                create_sample_data
                ;;
        esac
    else
        log_info "发现 ${CATEGORIES} 个数据集类别"
        ls "${DATA_DIR}"
    fi
}

# 创建示例数据集（用于测试）
create_sample_data() {
    log_info "创建示例数据集用于测试..."
    mkdir -p "${DATA_DIR}/bottle/train/good"
    mkdir -p "${DATA_DIR}/bottle/test/good"
    mkdir -p "${DATA_DIR}/bottle/test/broken"
    mkdir -p "${DATA_DIR}/bottle/ground_truth/broken"

    # 创建简单的测试图像
    for i in $(seq 1 20); do
        # 使用Python生成测试图像
        python -c "
from PIL import Image
import random
img = Image.new('RGB', (224, 224), color=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
img.save('${DATA_DIR}/bottle/train/good/${i}.png')
img.save('${DATA_DIR}/bottle/test/good/${i}.png')
"
    done

    log_info "示例数据集创建完成"
}

# 配置环境变量
configure_env() {
    log_info "配置环境变量..."

    # 创建 .env 文件
    cat > .env << EOF
# InduDet-Search 环境配置
export CUDA_VISIBLE_DEVICES=${NUM_GPUS}
export MAX_TRIALS=200
export SAVE_DIR=${RESULTS_DIR}
export DATA_DIR=${DATA_DIR}

# MiniMax API Key (可选)
# export MINIMAX_API_KEY="your-api-key"
EOF

    log_info "环境配置已保存到 .env"
}

# 运行测试
run_test() {
    log_info "运行快速测试..."

    source venv/bin/activate

    # 测试导入
    python -c "
from src.search import run_search
from src.models import AnomalyDetector
from src.data import MVTecDataset
print('✓ 所有模块导入成功')
"

    if [ $? -eq 0 ]; then
        log_info "模块测试通过!"
    else
        log_error "模块测试失败"
        exit 1
    fi
}

# 启动搜索任务
start_search() {
    log_info "启动架构搜索任务..."

    source venv/bin/activate

    # 使用screen或nohup后台运行
    if command -v screen &> /dev/null; then
        screen -dmS indudet python run_search.py \
            --n-trials 200 \
            --category bottle \
            --save-dir ${RESULTS_DIR} \
            --device cuda

        log_info "任务已启动在 screen 会话 'indudet' 中"
        log_info "使用 'screen -r indudet' 查看进度"
    else
        log_warn "screen 未安装，使用 nohup 运行..."
        nohup python run_search.py \
            --n-trials 200 \
            --category bottle \
            --save-dir ${RESULTS_DIR} \
            --device cuda \
            > ${RESULTS_DIR}/search.log 2>&1 &

        log_info "任务已启动，PID: $!"
        log_info "日志保存在: ${RESULTS_DIR}/search.log"
    fi
}

# 主函数
main() {
    echo ""
    echo "请选择操作:"
    echo "1. 安装依赖"
    echo "2. 准备数据"
    echo "3. 运行测试"
    echo "4. 启动搜索"
    echo "5. 完整部署 (1+2+3+4)"
    echo "0. 退出"
    echo ""
    read -p "请选择 [0-5]: " choice

    case $choice in
        1)
            check_system
            install_dependencies
            ;;
        2)
            prepare_data
            ;;
        3)
            run_test
            ;;
        4)
            start_search
            ;;
        5)
            check_system
            install_dependencies
            prepare_data
            configure_env
            run_test
            log_info "部署完成! 使用选项4启动搜索"
            ;;
        0)
            exit 0
            ;;
        *)
            log_error "无效选择"
            exit 1
            ;;
    esac
}

main
