"""多跳检索模块"""
from typing import List, Dict, Tuple
import logging
from vector_store import VectorStore
from embedder import Embedder

logger = logging.getLogger(__name__)

class MultiHopRetriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder,
                 top_k: int = 5, max_hops: int = 3, similarity_threshold: float = 0.3):
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.max_hops = max_hops
        self.similarity_threshold = similarity_threshold

    def retrieve_for_subqueries(self, subqueries: List[Dict]) -> Dict:
        """为所有子查询执行检索"""
        logger.info(f"Retrieving for {len(subqueries)} subqueries...")

        all_results = {
            'subquery_results': [],
            'total_evidence': [],
            'retrieval_stats': {
                'total_retrievals': 0,
                'successful_retrievals': 0,
                'avg_similarity': 0.0
            }
        }

        all_similarities = []

        for subquery_info in subqueries:
            subquery = subquery_info['subquery']
            logger.info(f"\nProcessing subquery {subquery_info['order']}: {subquery}")

            # 执行多跳检索
            hop_results = self._multi_hop_search(subquery, subquery_info)

            # 收集结果
            subquery_result = {
                'subquery': subquery,
                'order': subquery_info['order'],
                'type': subquery_info['type'],
                'hops': hop_results['hops'],
                'evidence': hop_results['evidence'],
                'stats': hop_results['stats']
            }

            all_results['subquery_results'].append(subquery_result)
            all_results['total_evidence'].extend(hop_results['evidence'])
            all_results['retrieval_stats']['total_retrievals'] += hop_results['stats']['total_retrievals']

            if hop_results['evidence']:
                all_results['retrieval_stats']['successful_retrievals'] += 1
                all_similarities.extend([e['similarity'] for e in hop_results['evidence']])

        # 计算平均相似度
        if all_similarities:
            all_results['retrieval_stats']['avg_similarity'] = sum(all_similarities) / len(all_similarities)

        # 去重证据
        all_results['total_evidence'] = self._deduplicate_evidence(all_results['total_evidence'])

        logger.info(f"\nTotal unique evidence collected: {len(all_results['total_evidence'])}")
        logger.info(f"Successful retrievals: {all_results['retrieval_stats']['successful_retrievals']}/{len(subqueries)}")

        return all_results

    def _multi_hop_search(self, query: str, subquery_info: Dict) -> Dict:
        """执行多跳检索"""
        result = {
            'hops': [],
            'evidence': [],
            'stats': {
                'total_retrievals': 0,
                'hops_performed': 0
            }
        }

        current_query = query
        retrieved_passages = set()

        for hop in range(self.max_hops):
            logger.info(f"  Hop {hop + 1}: {current_query[:100]}...")

            # 嵌入当前查询
            query_embedding = self.embedder.embed_single(current_query)

            # 检索
            search_results = self.vector_store.search(query_embedding, self.top_k)
            result['stats']['total_retrievals'] += 1

            # 过滤相似度阈值
            filtered_results = [
                (passage, sim) for passage, sim in search_results
                if sim >= self.similarity_threshold
            ]

            if not filtered_results:
                logger.info(f"  No results above threshold at hop {hop + 1}")
                break

            # 记录这一跳的结果
            hop_evidence = []
            for passage, similarity in filtered_results:
                passage_id = f"{passage.get('question_id', '')}_{passage.get('passage_id', '')}"

                if passage_id not in retrieved_passages:
                    retrieved_passages.add(passage_id)
                    evidence_item = {
                        'text': passage['text'],
                        'similarity': similarity,
                        'hop': hop + 1,
                        'source_query': current_query
                    }
                    hop_evidence.append(evidence_item)
                    result['evidence'].append(evidence_item)

            result['hops'].append({
                'hop_number': hop + 1,
                'query': current_query,
                'results_count': len(hop_evidence)
            })
            result['stats']['hops_performed'] += 1

            logger.info(f"  Found {len(hop_evidence)} new passages at hop {hop + 1}")

            # 如果第一跳就找到足够的证据,可以提前停止
            if hop == 0 and len(hop_evidence) >= 3:
                break

            # 为下一跳生成查询 (使用最相关的段落)
            if hop < self.max_hops - 1 and hop_evidence:
                # 简化实现:使用最相关段落的部分内容作为下一跳的查询
                top_passage = hop_evidence[0]['text']
                # 提取前50个词作为上下文扩展
                words = top_passage.split()[:50]
                current_query = query + " " + " ".join(words)
            else:
                break

        return result

    def _deduplicate_evidence(self, evidence_list: List[Dict]) -> List[Dict]:
        """去重证据"""
        seen_texts = set()
        unique_evidence = []

        for evidence in evidence_list:
            text_hash = hash(evidence['text'])
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_evidence.append(evidence)

        return unique_evidence