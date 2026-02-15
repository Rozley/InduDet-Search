#!/usr/bin/env python3
"""
InduDet-Search 主入口脚本

用法:
    # 单类别搜索
    python run_search.py --category bottle --n-trials 200

    # 多类别搜索
    python run_search.py --categories bottle,cable,transistor --n-trials 100

    # 所有类别搜索
    python run_search.py --all-categories --n-trials 200

    # 快速测试
    python run_search.py --n-trials 10 --verbose
"""

import argparse
import sys

from src.search import run_search


# MVTec AD 所有类别
MVTEC_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'guitar', 'hazelnut', 'leather', 'metal_nut', 'pill',
    'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]


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
    python run_search.py --data-dir ./data --category bottle --n-trials 100

    # 多类别搜索
    python run_search.py --categories bottle,cable,transistor --n-trials 100

    # 所有15个类别搜索
    python run_search.py --all-categories --n-trials 200

    # 快速测试（10次搜索）
    python run_search.py --n-trials 10 --verbose

搜索策略:
    --strategy sequential  # 顺序搜索每个类别
    --strategy joint      # 联合搜索（推荐）
        """
    )

    # 配置文件
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='配置文件路径 (default: configs/config.yaml)'
    )

    # 数据集 - 支持单类别或多类别
    parser.add_argument(
        '--category',
        type=str,
        default=None,
        help='单个物体类别 (与 --categories 二选一)'
    )
    parser.add_argument(
        '--categories',
        type=str,
        default=None,
        help='多个类别，逗号分隔 (与 --category 二选一)'
    )
    parser.add_argument(
        '--all-categories',
        action='store_true',
        help='使用所有MVTec AD类别'
    )

    # 搜索参数
    parser.add_argument(
        '--n-trials',
        type=int,
        default=200,
        help='每个类别的搜索次数 (default: 200)'
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
    parser.add_argument(
        '--strategy',
        type=str,
        default='joint',
        choices=['sequential', 'joint'],
        help='搜索策略: sequential(顺序) 或 joint(联合搜索) (default: joint)'
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


def resolve_categories(args) -> list:
    """解析并确定要搜索的类别列表"""
    if args.all_categories:
        return MVTEC_CATEGORIES
    elif args.categories:
        return [c.strip() for c in args.categories.split(',')]
    elif args.category:
        return [args.category]
    else:
        # 默认使用 bottle
        return ['bottle']


def main():
    """主函数"""
    args = parse_args()

    # 解析类别
    categories = resolve_categories(args)

    print("=" * 60)
    print("InduDet-Search")
    print("增量式工业异常检测架构搜索系统")
    print("=" * 60)

    if len(categories) == 1:
        print(f"类别: {categories[0]}")
    else:
        print(f"类别: {len(categories)} 个 ({', '.join(categories[:5])}{'...' if len(categories) > 5 else ''})")

    print(f"搜索策略: {args.strategy}")
    print(f"搜索次数: {args.n_trials}/类别")
    print(f"图像大小: {args.image_size}")
    print(f"批次大小: {args.batch_size}")
    print(f"设备: {args.device}")
    print("=" * 60)

    try:
        results, best_config, best_score = run_search(
            config_path=args.config,
            data_dir=args.data_dir if hasattr(args, 'data_dir') else './data/mvtec',
            categories=categories,
            n_trials=args.n_trials,
            image_size=args.image_size,
            batch_size=args.batch_size,
            strategy=args.strategy,
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
