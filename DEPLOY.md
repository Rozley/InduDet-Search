# 快速部署指南

## 方式一：直接运行 (Linux/Mac)

```bash
# 1. 上传项目到服务器
scp -r InduDet-Search/ user@your-server:/path/to/

# 2. 连接服务器
ssh user@your-server

# 3. 进入项目目录
cd /path/to/InduDet-Search

# 4. 运行部署脚本
bash deploy.sh
```

## 方式二：使用 Docker (推荐)

### 1. 安装 Docker 和 NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
curl https://get.docker.com | sh
sudo systemctl start docker

# 安装 NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. 构建并运行

```bash
# GPU版本 (推荐)
docker compose up indudet-gpu -d

# 查看日志
docker compose logs -f indudet-gpu

# CPU版本
docker compose up indudet-cpu -d
```

## 方式三：租用云GPU服务器

### 推荐平台

| 平台 | 特点 | 价格 |
|------|------|------|
| [AutoDL](https://www.autodl.com) | 国内访问快， RTX 4090 充足 | ¥2-4/小时 |
| [智星云](https://www.cloudgalaxy.cn) | RTX 5090/4090 | ¥3-6/小时 |
| [阿里云](https://www.aliyun.com) | 稳定可靠，售后好 | ¥5-10/小时 |
| [腾讯云](https://cloud.tencent.com) | GPU实例丰富 | ¥4-8/小时 |

### 租用步骤 (以 AutoDL 为例)

1. **注册账号** - 使用学生/教育邮箱有优惠
2. **租用实例**
   - 选择 GPU: RTX 4090 (24GB) 或 A100
   - 选择镜像: Ubuntu 20.04 + CUDA 11.8
   - 开启 Jupyter 或 SSH
3. **上传数据**
   ```bash
   # 使用 AutoDL 助手或 SCP
   scp -r InduDet-Search/ root@your-instance:/root/
   ```
4. **运行**
   ```bash
   ssh root@your-instance
   cd InduDet-Search
   pip install -r requirements.txt
   python run_search.py --n-trials 200
   ```

## 方式四：VS Code 远程开发

1. **安装 Remote-SSH 扩展**
2. **连接服务器**
   - 按 `F1` → `Remote-SSH: Connect to Host`
   - 输入 `user@your-server`
3. **在服务器上运行**
   - 打开项目文件夹
   - 安装依赖
   - 运行 `python run_search.py`

## 快速检查清单

部署前确认：

- [ ] Python 3.8+ 已安装
- [ ] GPU 驱动版本 >= 515.65
- [ ] CUDA 11.8+ 已安装
- [ ] MVTec AD 数据集已上传
- [ ] 依赖包已安装
- [ ] 运行测试通过

## 常用命令

```bash
# 查看GPU状态
nvidia-smi

# 查看GPU进程
ps aux | grep python

# 停止所有Python进程
pkill -f python

# 后台运行 (Linux)
nohup python run_search.py > output.log 2>&1 &

# 查看运行日志
tail -f output.log
```

## 常见问题

### Q: CUDA out of memory
A: 减小 `batch_size` 或使用较小的 backbone

### Q: 进程被杀掉
A: 检查内存使用，可能需要增加 swap

### Q: 下载依赖超时
A: 使用国内镜像源
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
