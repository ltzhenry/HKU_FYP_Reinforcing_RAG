"""子查询分解模块"""
from typing import List, Dict
import re
import logging

logger = logging.getLogger(__name__)

class QueryDecomposer:
    def __init__(self, max_subqueries: int = 4):
        self.max_subqueries = max_subqueries

    def decompose(self, analysis_result: Dict) -> List[Dict]:
        """将复杂问题分解为子查询"""
        question = analysis_result['question']

        if not analysis_result['requires_decomposition']:
            # 简单问题不需要分解
            return [{
                'subquery': question,
                'type': 'direct',
                'order': 1,
                'dependency': None
            }]

        logger.info(f"Decomposing complex question: {question}")

        subqueries = []

        # 策略1: 基于连接词分割
        if analysis_result['features']['has_conjunctions']:
            subqueries.extend(self._split_by_conjunctions(question))

        # 策略2: 基于逗号和分号分割
        if analysis_result['features']['has_multiple_clauses'] and len(subqueries) == 0:
            subqueries.extend(self._split_by_punctuation(question))

        # 策略3: 提取核心概念查询
        if len(subqueries) == 0:
            subqueries.extend(self._extract_core_concepts(question))

        # 如果所有策略都失败,返回原问题
        if len(subqueries) == 0:
            subqueries = [{
                'subquery': question,
                'type': 'direct',
                'order': 1,
                'dependency': None
            }]

        # 限制子查询数量
        subqueries = subqueries[:self.max_subqueries]

        logger.info(f"Generated {len(subqueries)} subqueries")
        for i, sq in enumerate(subqueries, 1):
            logger.info(f"  Subquery {i}: {sq['subquery']}")

        return subqueries

    def _split_by_conjunctions(self, question: str) -> List[Dict]:
        """基于连接词分割问题"""
        subqueries = []

        # 分割模式
        patterns = [r'\band\b', r'\bor\b', r'\bas well as\b']

        for pattern in patterns:
            parts = re.split(pattern, question, flags=re.IGNORECASE)
            if len(parts) > 1:
                for i, part in enumerate(parts, 1):
                    part = part.strip()
                    if part and len(part) > 5:  # 过滤太短的片段
                        # 确保每个部分都是完整的问题
                        if not part.endswith('?'):
                            part += '?'

                        subqueries.append({
                            'subquery': part,
                            'type': 'conjunction_split',
                            'order': i,
                            'dependency': None if i == 1 else i - 1
                        })
                break

        return subqueries

    def _split_by_punctuation(self, question: str) -> List[Dict]:
        """基于标点符号分割问题"""
        subqueries = []
        parts = re.split(r'[,;]', question)

        if len(parts) > 1:
            for i, part in enumerate(parts, 1):
                part = part.strip()
                if part and len(part) > 5:
                    if not part.endswith('?'):
                        part += '?'

                    subqueries.append({
                        'subquery': part,
                        'type': 'punctuation_split',
                        'order': i,
                        'dependency': None if i == 1 else i - 1
                    })

        return subqueries

    def _extract_core_concepts(self, question: str) -> List[Dict]:
        """提取核心概念生成子查询"""
        subqueries = []

        # 生成一个概念提取查询和一个关系查询
        # 这是一个简化的实现,实际可以使用NLP工具

        # 主查询
        subqueries.append({
            'subquery': question,
            'type': 'main_query',
            'order': 1,
            'dependency': None
        })

        # 如果问题包含"how"或"why",添加机制查询
        if any(word in question.lower() for word in ['how', 'why', 'mechanism']):
            mechanism_query = question.replace('?', ' mechanism?')
            subqueries.append({
                'subquery': mechanism_query,
                'type': 'mechanism_query',
                'order': 2,
                'dependency': 1
            })

        return subqueries