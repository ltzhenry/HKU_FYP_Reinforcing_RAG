"""答案生成模块"""
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class AnswerGenerator:
    def __init__(self):
        pass

    def generate(self, question: str, integrated_evidence: Dict,
                 analysis_result: Dict, retrieval_results: Dict) -> Dict:
        """基于证据生成答案"""
        logger.info("Generating answer...")

        evidence_list = integrated_evidence['evidence']

        if not evidence_list:
            return {
                'answer': "No sufficient evidence found to answer this question.",
                'confidence': 0.0,
                'reasoning_path': [],
                'sources': []
            }

        # 构建推理路径
        reasoning_path = self._build_reasoning_path(
            question,
            analysis_result,
            retrieval_results['subquery_results'],
            evidence_list
        )

        # 生成答案摘要
        answer = self._synthesize_answer(question, evidence_list)

        # 提取来源
        sources = self._extract_sources(evidence_list)

        # 计算置信度
        confidence = integrated_evidence['validation']['avg_confidence']

        result = {
            'answer': answer,
            'confidence': confidence,
            'reasoning_path': reasoning_path,
            'sources': sources,
            'evidence_count': len(evidence_list)
        }

        logger.info(f"Answer generated with confidence: {confidence:.3f}")

        return result

    def _build_reasoning_path(self, question: str, analysis: Dict,
                             subquery_results: List[Dict], evidence: List[Dict]) -> List[Dict]:
        """构建推理路径"""
        path = []

        # 第一步:问题分析
        path.append({
            'step': 1,
            'type': 'analysis',
            'description': f"Analyzed question complexity: {analysis['complexity_score']:.2f}",
            'is_complex': analysis['is_complex']
        })

        # 第二步:查询分解(如果需要)
        if analysis['requires_decomposition']:
            path.append({
                'step': 2,
                'type': 'decomposition',
                'description': f"Decomposed into {len(subquery_results)} sub-queries",
                'subqueries': [sq['subquery'] for sq in subquery_results]
            })

        # 第三步:多跳检索
        total_hops = sum(sq['stats']['hops_performed'] for sq in subquery_results)
        path.append({
            'step': 3,
            'type': 'retrieval',
            'description': f"Performed multi-hop retrieval ({total_hops} total hops)",
            'evidence_found': len(evidence)
        })

        # 第四步:证据整合
        path.append({
            'step': 4,
            'type': 'integration',
            'description': f"Integrated and validated {len(evidence)} evidence pieces",
            'top_evidence_scores': [f"{e['quality_score']:.3f}" for e in evidence[:3]]
        })

        # 第五步:答案生成
        path.append({
            'step': 5,
            'type': 'generation',
            'description': "Synthesized final answer from evidence"
        })

        return path

    def _synthesize_answer(self, question: str, evidence_list: List[Dict]) -> str:
        """合成答案 (简化版本,实际应使用生成模型)"""
        # 这里使用一个简单的基于证据的答案生成
        # 实际项目中应该使用T5, GPT等生成模型

        if not evidence_list:
            return "No answer available."

        # 选择最相关的证据片段
        top_evidence = evidence_list[0]['text']

        # 提取关键句子作为答案
        sentences = top_evidence.split('.')
        answer_sentences = sentences[:2]  # 取前两句

        answer = '. '.join(s.strip() for s in answer_sentences if s.strip())
        if answer and not answer.endswith('.'):
            answer += '.'

        # 如果有多个高质量证据,添加补充信息
        if len(evidence_list) > 1 and evidence_list[1]['quality_score'] > 0.6:
            additional = evidence_list[1]['text'].split('.')[0].strip()
            if additional and additional not in answer:
                answer += f" Additionally, {additional.lower()}."

        return answer

    def _extract_sources(self, evidence_list: List[Dict]) -> List[Dict]:
        """提取证据来源"""
        sources = []
        for i, evidence in enumerate(evidence_list, 1):
            sources.append({
                'rank': i,
                'text': evidence['text'][:200] + '...' if len(evidence['text']) > 200 else evidence['text'],
                'similarity': evidence['similarity'],
                'quality_score': evidence['quality_score'],
                'hop': evidence['hop']
            })
        return sources