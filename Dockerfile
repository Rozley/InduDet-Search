# InduDet-Search Dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu22.04

LABEL maintainer="user@example.com"
LABEL description="InduDet-Search: 增量式工业异常检测架构搜索系统"

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-venv \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# 创建虚拟环境
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 创建数据目录
RUN mkdir -p /app/data /app/results

# 设置默认命令
CMD ["python", "run_search.py", "--n-trials", "200", "--category", "bottle"]

# 暴露端口
EXPOSE 8888
