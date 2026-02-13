#!/usr/bin/env python3
"""
InduDet-Search 主入口脚本

用法:
    python run_search.py --config configs/config.yaml --n-trials 200
    python run_search.py --data-dir ./data --category bottle --n-trials 100
"""

import argparse
import sys

from src.search import run_search


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='InduDet-Search: 增量式工业异常检测架构搜索系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 使用默认配置搜索
    python run_search.py --n-trials 200

    # 指定数据集和类别
    python run_search.py --data-dir ./data/mvtec --category bottle --n-trials 100

    # 快速测试（10次搜索）
    python run_search.py --n-trials 10 --verbose
        """
    )

    # 配置文件
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='配置文件路径 (default: configs/config.yaml)'
    )

    # 数据集
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data/mvtec',
        help='数据集目录 (default: ./data/mvtec)'
    )
    parser.add_argument(
        '--category',
        type=str,
        default='bottle',
        help='物体类别 (default: bottle)'
    )

    # 搜索参数
    parser.add_argument(
        '--n-trials',
        type=int,
        default=200,
        help='搜索次数 (default: 200)'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='图像大小 (default: 224)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='批次大小 (default: 32)'
    )

    # 输出
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./results',
        help='结果保存目录 (default: ./results)'
    )

    # 设备
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='计算设备: cuda/cpu (default: cuda)'
    )

    # 日志
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='详细输出'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    print("=" * 60)
    print("InduDet-Search")
    print("增量式工业异常检测架构搜索系统")
    print("=" * 60)
    print(f"配置: {args.config}")
    print(f"数据集: {args.data_dir}/{args.category}")
    print(f"搜索次数: {args.n_trials}")
    print(f"图像大小: {args.image_size}")
    print(f"批次大小: {args.batch_size}")
    print(f"设备: {args.device}")
    print("=" * 60)

    try:
        results, best_config, best_score = run_search(
            config_path=args.config,
            data_dir=args.data_dir,
            category=args.category,
            n_trials=args.n_trials,
            image_size=args.image_size,
            batch_size=args.batch_size,
            save_dir=args.save_dir,
            device=args.device,
            verbose=args.verbose,
        )

        print("\n搜索完成!")
        print(f"最佳AUROC: {best_score:.4f}")
        print(f"最佳配置: {best_config}")

        return 0

    except KeyboardInterrupt:
        print("\n用户中断搜索")
        return 130

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
