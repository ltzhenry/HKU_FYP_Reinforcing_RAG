"""数据加载模块"""
import logging
from datasets import load_dataset
from typing import List, Dict
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        self.dataset = None
        self.train_data = None
        self.test_data = None

    def load_bioasq_dataset(self, train_ratio: float = 0.8):
        """加载BioASQ数据集并手动分割训练/测试集"""
        logger.info("Loading BioASQ dataset from HuggingFace...")

        try:
            # 加载数据集
            ds = load_dataset("enelpol/rag-mini-bioasq", "question-answer-passages")
            # BioASQ数据集通常只有一个split，我们需要手动分割
            # 检查可用的splits
            available_splits = list(ds.keys())
            logger.info(f"Available splits: {available_splits}")

            # 使用第一个可用的split
            main_split = available_splits[0]
            full_dataset = ds[main_split]

            logger.info(f"Loaded {len(full_dataset)} examples from '{main_split}' split")

            # 手动分割数据集
            indices = list(range(len(full_dataset)))
            random.shuffle(indices)

            split_point = int(len(indices) * train_ratio)
            train_indices = indices[:split_point]
            test_indices = indices[split_point:]

            # 创建训练和测试数据
            self.train_data = [full_dataset[i] for i in train_indices]
            self.test_data = [full_dataset[i] for i in test_indices]

            logger.info(f"Dataset split: Train size: {len(self.train_data)}, Test size: {len(self.test_data)}")

            # 保存完整数据集引用
            self.dataset = ds

            # 打印样本以检查数据结构
            logger.info("\nSample data structure:")
            if self.train_data:
                sample = self.train_data[0]
                logger.info(f"Keys: {sample.keys()}")
                logger.info(f"Question: {sample.get('question', 'N/A')[:100]}...")
                logger.info(f"Answer: {sample.get('answer', 'N/A')[:100]}...")

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            logger.info("Creating empty dataset placeholders...")
            self.train_data = []
            self.test_data = []

    def get_passages(self, split: str = 'train', max_passages: int = None) -> List[Dict]:
        """从数据集中提取段落用于构建索引"""
        data = self.train_data if split == 'train' else self.test_data

        if not data:
            logger.warning(f"No data available for split '{split}'")
            return []

        passages = []
        passage_id = 0

        for item in data:
            # 从答案中提取段落
            answer = item.get('answer', '')
            if answer and len(answer.strip()) > 0:
                passages.append({
                    'id': passage_id,
                    'text': answer,
                    'source': 'answer',
                    'question_id': item.get('id', -1)
                })
                passage_id += 1

            # 如果有相关段落ID，也可以处理
            # relevant_passages = item.get('relevant_passage_ids', [])
            # 注意：这里的relevant_passage_ids可能是字符串，需要解析

            if max_passages and len(passages) >= max_passages:
                break

        logger.info(f"Extracted {len(passages)} passages from {split} split")
        return passages

    def get_questions(self, split: str = 'test', max_questions: int = None) -> List[Dict]:
        """获取问题用于查询"""
        data = self.train_data if split == 'train' else self.test_data

        if not data:
            logger.warning(f"No data available for split '{split}'")
            return []

        questions = []

        for item in data:
            question = item.get('question', '')
            answer = item.get('answer', '')

            if question:
                # 将答案转换为列表格式以保持一致性
                answers = [answer] if answer else []

                questions.append({
                    'id': item.get('id', -1),
                    'question': question,
                    'answers': answers,
                    'relevant_passage_ids': item.get('relevant_passage_ids', [])
                })

            if max_questions and len(questions) >= max_questions:
                break

        logger.info(f"Extracted {len(questions)} questions from {split} split")
        return questions