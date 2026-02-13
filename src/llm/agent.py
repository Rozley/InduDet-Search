"""
LLM Agent模块
使用MiniMax M2.1 API生成架构建议
"""

import json
import os
from typing import Any, Dict, List, Optional

import requests


class LLMClientBase:
    """LLM客户端基类"""
    def generate(self, prompt: str, **kwargs) -> Dict:
        raise NotImplementedError


class MiniMaxM2Client(LLMClientBase):
    """
    MiniMax M2.1 API 客户端

    优势:
    - 国内访问延迟低
    - 支持长上下文 (128K)
    - 成本低于GPT-4
    - 中文优化
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = "https://api.minimax.chat/v1",
        model: str = "MiniMax-M2.1",
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ):
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, prompt: str, **kwargs) -> Dict:
        """生成响应"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "temperature": kwargs.get('temperature', self.temperature),
        }

        try:
            response = requests.post(
                f"{self.base_url}/text/chatcompletion_v2",
                headers=headers,
                json=data,
                timeout=60,
            )

            response.raise_for_status()
            result = response.json()

            return {
                'content': result['choices'][0]['message']['content'],
                'usage': result.get('usage', {}),
                'model': self.model,
                'status': 'success',
            }

        except requests.exceptions.RequestException as e:
            return {
                'content': '',
                'error': str(e),
                'status': 'error',
            }

    def generate_json(self, prompt: str, **kwargs) -> Dict:
        """生成JSON格式响应"""
        # 添加JSON格式要求到prompt
        json_prompt = f"""{prompt}

请严格按照以下JSON格式输出（不要添加任何其他内容）：
{{
    "backbone": "网络骨架名称",
    "feature_levels": "特征层级",
    "method": "检测方法",
    "memory_size": 记忆库大小,
    "k": k值,
    "reasoning": "设计理由"
}}
"""
        result = self.generate(json_prompt, **kwargs)

        if result['status'] == 'error':
            return result

        # 尝试解析JSON
        try:
            # 提取JSON内容
            content = result['content']
            # 清理markdown代码块
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()

            json_data = json.loads(content)
            result['parsed'] = json_data
            return result

        except json.JSONDecodeError as e:
            return {
                'content': result['content'],
                'error': f"JSON解析失败: {e}",
                'status': 'error',
            }


class SimpleLLMAgent:
    """
    简化版LLM Agent

    功能:
    1. 基于当前搜索状态生成架构建议
    2. 分析失败原因
    3. 提供设计洞察
    """

    def __init__(
        self,
        llm_client: Optional[LLMClientBase] = None,
        use_for_suggestion: bool = True,
    ):
        """
        Args:
            llm_client: LLM客户端实例
            use_for_suggestion: 是否使用LLM生成建议
        """
        self.llm = llm_client or MiniMaxM2Client()
        self.use_for_suggestion = use_for_suggestion

    def suggest_architecture(
        self,
        task_spec: Optional[Dict] = None,
        history_summary: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        基于任务规格和历史结果生成架构建议

        Args:
            task_spec: 任务规格
            history_summary: 历史最佳结果摘要

        Returns:
            {
                'config': Dict,
                'reasoning': str,
                'source': 'llm' or 'default',
            }
        """
        if not self.use_for_suggestion or not self.llm:
            # 返回默认配置
            return self._get_default_config()

        # 构建Prompt
        prompt = self._build_suggestion_prompt(task_spec, history_summary)

        # 调用LLM
        result = self.llm.generate_json(prompt)

        if result['status'] == 'error':
            print(f"LLM调用失败: {result.get('error', '未知错误')}")
            return self._get_default_config()

        # 解析结果
        try:
            parsed = result.get('parsed', {})

            # 验证配置完整性
            config = {
                'backbone': parsed.get('backbone', 'ResNet50'),
                'feature_levels': parsed.get('feature_levels', 'L2+L3'),
                'method': parsed.get('method', 'memory_bank'),
                'memory_size': int(parsed.get('memory_size', 1000)),
                'k': int(parsed.get('k', 9)),
            }

            return {
                'config': config,
                'reasoning': parsed.get('reasoning', '基于LLM分析'),
                'source': 'llm',
            }

        except Exception as e:
            print(f"解析LLM响应失败: {e}")
            return self._get_default_config()

    def _build_suggestion_prompt(
        self,
        task_spec: Optional[Dict],
        history_summary: Optional[Dict],
    ) -> str:
        """构建建议Prompt"""
        prompt = f"""你是一个工业异常检测模型架构专家。请为以下任务设计最优的模型架构。

任务要求:
- 数据集: {task_spec.get('dataset', 'MVTec AD') if task_spec else 'MVTec AD'}
- 部署设备: {task_spec.get('target_device', 'GPU') if task_spec else 'GPU'}
- 延迟约束: {task_spec.get('latency_constraint', '100ms') if task_spec else '100ms'}

搜索空间选项:
- backbone: ResNet18, ResNet50, EfficientNet-B0, MobileNetV3, ViT-Small
- feature_levels: L2, L2+L3, L2+L3+L4
- method: memory_bank, distribution, contrastive
- memory_size: 500, 1000, 2000
- k: 1, 5, 9

历史最佳结果:"""

        if history_summary:
            prompt += f"""
- 最高AUROC: {history_summary.get('best_score', '未知')}
- 常用Backbone: {history_summary.get('common_backbone', 'ResNet50')}
"""

        prompt += """
请根据以上信息，推荐一个最优的架构配置。
"""

        return prompt

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'config': {
                'backbone': 'ResNet50',
                'feature_levels': 'L2+L3',
                'method': 'memory_bank',
                'memory_size': 1000,
                'k': 9,
            },
            'reasoning': '使用默认配置',
            'source': 'default',
        }

    def analyze_results(self, results: List[Dict]) -> Dict[str, Any]:
        """分析搜索结果"""
        if not results:
            return {'insights': '没有足够的结果进行分析'}

        # 按AUROC排序
        sorted_results = sorted(results, key=lambda x: x.get('auroc', 0), reverse=True)

        # 统计成功模式
        successful = [r for r in sorted_results if r.get('auroc', 0) > 0.85]
        failed = [r for r in sorted_results if r.get('auroc', 0) < 0.6]

        insights = {
            'n_total': len(results),
            'n_successful': len(successful),
            'n_failed': len(failed),
            'best_config': sorted_results[0].get('config', {}) if sorted_results else {},
            'best_score': sorted_results[0].get('auroc', 0) if sorted_results else 0,
        }

        # 分析成功配置的模式
        if successful:
            backbones = [r['config'].get('backbone') for r in successful]
            from collections import Counter
            most_common_backbone = Counter(backbones).most_common(1)
            insights['recommended_backbone'] = most_common_backbone[0][0] if most_common_backbone else 'ResNet50'

        return insights


def create_llm_client(
    provider: str = 'minimax',
    api_key: str = None,
    **kwargs,
) -> LLMClientBase:
    """创建LLM客户端的工厂函数"""
    if provider == 'minimax':
        return MiniMaxM2Client(api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def create_llm_agent(
    llm_client: Optional[LLMClientBase] = None,
    use_for_suggestion: bool = True,
) -> SimpleLLMAgent:
    """创建LLM Agent的便捷函数"""
    return SimpleLLMAgent(
        llm_client=llm_client,
        use_for_suggestion=use_for_suggestion,
    )
