"""答案生成模块 - 使用 DeepSeek LLM"""
from typing import Dict, List
import logging
import os
from openai import OpenAI

logger = logging.getLogger(__name__)

class AnswerGenerator:
    def __init__(self):
        # 初始化 DeepSeek 客户端
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            logger.warning("DEEPSEEK_API_KEY not found in environment. Using simple synthesis.")
            self.client = None
        else:
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )

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

        # 生成答案
        if self.client:
            try:
                answer = self._synthesize_answer_with_llm(question, evidence_list, analysis_result)
            except Exception as e:
                logger.error(f"LLM answer generation failed: {e}. Using simple synthesis.")
                answer = self._synthesize_answer_simple(question, evidence_list)
        else:
            answer = self._synthesize_answer_simple(question, evidence_list)

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

    def _synthesize_answer_with_llm(self, question: str, evidence_list: List[Dict],
                                     analysis_result: Dict) -> str:
        """使用 DeepSeek LLM 合成答案"""

        # 准备证据文本
        evidence_texts = []
        for i, evidence in enumerate(evidence_list[:5], 1):  # 最多使用前5个证据
            evidence_texts.append(f"Evidence {i} (Quality: {evidence['quality_score']:.2f}):\n{evidence['text']}\n")

        evidence_context = "\n".join(evidence_texts)

        prompt = f"""You are an expert assistant that synthesizes information from multiple sources to answer questions accurately.

Question: {question}

Available Evidence:
{evidence_context}

Instructions:
1. Synthesize a comprehensive answer based on the provided evidence
2. Prioritize evidence with higher quality scores
3. Be concise but complete
4. If the evidence is insufficient or contradictory, acknowledge this
5. Use factual language and avoid speculation
6. generate answer after analysis of evidence, you can use your own words, make your words logically appropriate to answer the questions

Provide a clear, well-structured answer:"""

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a precise and factual answer synthesis assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )

        answer = response.choices[0].message.content.strip()

        logger.info(f"LLM generated answer: {answer[:100]}...")

        return answer

    def _synthesize_answer_simple(self, question: str, evidence_list: List[Dict]) -> str:
        """简单的基于证据的答案生成 (回退方案)"""

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