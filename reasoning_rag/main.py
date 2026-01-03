"""主运行文件"""
import argparse
import logging
import os
from config import Config
from data_loader import DataLoader
from reasoning_rag import ReasoningRAG
from evaluator import RAGEvaluator
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_index(rag_system: ReasoningRAG, data_loader: DataLoader, index_path: str, max_passages: int = 1000):
    """构建并保存索引"""
    logger.info("Building vector index from dataset...")

    # 加载数据集 (80% train, 20% test)
    data_loader.load_bioasq_dataset(train_ratio=0.8)

    # 获取训练集的段落（限制数量以加快演示）
    passages = data_loader.get_passages(split='train', max_passages=max_passages)

    if not passages:
        logger.error("No passages found! Cannot build index.")
        return

    # 构建索引
    rag_system.build_index(passages)

    # 保存索引
    rag_system.save_index(index_path)
    logger.info(f"Index saved to {index_path}")

def run_evaluation(rag_system: ReasoningRAG, data_loader: DataLoader, sample_size: int = 50):
    """运行评估测试"""
    logger.info(f"\n{'#'*80}")
    logger.info(f"# RUNNING EVALUATION TEST (Sample size: {sample_size})")
    logger.info(f"{'#'*80}\n")

    # 获取测试问题
    test_questions = data_loader.get_questions(split='test')

    # 随机采样
    if len(test_questions) > sample_size:
        test_questions = random.sample(test_questions, sample_size)

    logger.info(f"Evaluating on {len(test_questions)} test questions...\n")

    # 处理所有问题
    results = []
    for i, question_data in enumerate(test_questions, 1):
        logger.info(f"\n[{i}/{len(test_questions)}] Processing...")
        result = rag_system.query(question_data['question'], verbose=False)
        results.append(result)

    # 评估
    evaluator = RAGEvaluator()
    metrics = evaluator.evaluate_batch(results, test_questions)

    return metrics, results

def run_demo(rag_system: ReasoningRAG, data_loader: DataLoader, sample_size: int = 3):
    """运行演示对话"""
    logger.info(f"\n{'#'*80}")
    logger.info(f"# RUNNING DEMO CONVERSATIONS (Sample size: {sample_size})")
    logger.info(f"{'#'*80}\n")

    # 获取测试问题
    test_questions = data_loader.get_questions(split='test')

    # 随机选择几个问题
    demo_questions = random.sample(test_questions, min(sample_size, len(test_questions)))

    for i, question_data in enumerate(demo_questions, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"DEMO {i}/{len(demo_questions)}")
        logger.info(f"{'='*80}\n")

        question = question_data['question']
        ground_truth_answers = question_data['answers']

        # 显示问题
        print(f"\n🤔 QUESTION:")
        print(f"   {question}\n")

        # 处理问题
        result = rag_system.query(question, verbose=True)

        # 显示分析结果
        print(f"\n📊 ANALYSIS:")
        print(f"   Complexity Score: {result['analysis']['complexity_score']:.2f}")
        print(f"   Is Complex: {'Yes' if result['analysis']['is_complex'] else 'No'}")
        print(f"   Requires Decomposition: {'Yes' if result['analysis']['requires_decomposition'] else 'No'}")

        # 显示子查询
        if len(result['subqueries']) > 1:
            print(f"\n🔍 SUB-QUERIES ({len(result['subqueries'])}):")
            for j, sq in enumerate(result['subqueries'], 1):
                print(f"   {j}. [{sq['type']}] {sq['subquery']}")

        # 显示检索统计
        print(f"\n📚 RETRIEVAL STATS:")
        print(f"   Total Retrievals: {result['retrieval']['retrieval_stats']['total_retrievals']}")
        print(f"   Successful: {result['retrieval']['retrieval_stats']['successful_retrievals']}/{len(result['subqueries'])}")
        print(f"   Avg Similarity: {result['retrieval']['retrieval_stats']['avg_similarity']:.3f}")
        print(f"   Total Evidence: {len(result['evidence']['evidence'])}")

        # 显示推理路径
        print(f"\n🧠 REASONING PATH:")
        for step in result['answer']['reasoning_path']:
            print(f"   Step {step['step']} [{step['type']}]: {step['description']}")

        # 显示答案
        print(f"\n💡 GENERATED ANSWER:")
        print(f"   Confidence: {result['answer']['confidence']:.3f}")
        print(f"   {result['answer']['answer']}\n")

        # 显示证据来源
        if result['answer']['sources']:
            print(f"\n📖 EVIDENCE SOURCES (Top {min(3, len(result['answer']['sources']))}):")
            for source in result['answer']['sources'][:3]:
                print(f"\n   [{source['rank']}] Quality: {source['quality_score']:.3f} | Similarity: {source['similarity']:.3f} | Hop: {source['hop']}")
                print(f"       {source['text']}")

        # 显示Ground Truth (参考)
        print(f"\n✅ GROUND TRUTH ANSWERS (for reference):")
        for j, ans in enumerate(ground_truth_answers[:2], 1):
            print(f"   {j}. {ans}")

        print(f"\n{'='*80}\n")

        # 暂停以便查看
        if i < len(demo_questions):
            input("Press Enter to continue to next demo...\n")


def run_interactive(rag_system: ReasoningRAG):
    """运行交互式问答模式"""
    logger.info(f"\n{'#' * 80}")
    logger.info(f"# INTERACTIVE QUESTION ANSWERING MODE")
    logger.info(f"# Type 'quit' or 'exit' to stop")
    logger.info(f"{'#' * 80}\n")

    print("\n🤖 Welcome to Reasoning RAG Interactive Mode!")
    print("📝 You can ask any question and see the complete reasoning process.\n")

    question_count = 0

    while True:
        try:
            # 获取用户输入
            print(f"\n{'=' * 80}")
            user_input = input("\n💭 Your Question: ").strip()

            # 检查退出命令
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye! Thanks for using Reasoning RAG!\n")
                break

            # 检查空输入
            if not user_input:
                print("⚠️  Please enter a question.")
                continue

            question_count += 1

            logger.info(f"\n{'=' * 80}")
            logger.info(f"INTERACTIVE QUERY #{question_count}")
            logger.info(f"{'=' * 80}\n")

            # 处理问题
            result = rag_system.query(user_input, verbose=True)

            # 显示结果
            print(f"\n{'=' * 80}")
            print(f"📊 ANALYSIS RESULTS")
            print(f"{'=' * 80}\n")

            # 1. 复杂度分析
            print(f"🔍 Complexity Analysis:")
            print(f"   • Complexity Score: {result['analysis']['complexity_score']:.2f}")
            print(f"   • Classification: {'Complex' if result['analysis']['is_complex'] else 'Simple'}")
            print(f"   • Needs Decomposition: {'Yes' if result['analysis']['requires_decomposition'] else 'No'}")
            print(f"   • Reasoning Depth: {result['analysis']['reasoning_depth']}")

            # 2. 查询分解
            if len(result['subqueries']) > 1:
                print(f"\n🧩 Query Decomposition ({len(result['subqueries'])} sub-queries):")
                for i, sq in enumerate(result['subqueries'], 1):
                    print(f"   {i}. [{sq['type'].upper()}] {sq['subquery']}")
                    if sq.get('dependencies'):
                        print(f"      Dependencies: {sq['dependencies']}")
            else:
                print(f"\n🧩 Query Type: Single query (no decomposition needed)")

            # 3. 检索统计
            print(f"\n📚 Retrieval Statistics:")
            stats = result['retrieval']['retrieval_stats']
            print(f"   • Total Retrievals: {stats['total_retrievals']}")
            print(f"   • Successful: {stats['successful_retrievals']}/{len(result['subqueries'])}")
            print(f"   • Average Similarity: {stats['avg_similarity']:.3f}")
            print(f"   • Total Evidence Collected: {len(result['evidence']['evidence'])}")
            print(f"   • High Quality Evidence: {stats.get('high_quality_count', 0)}")

            # 4. 推理路径
            print(f"\n🧠 Reasoning Path:")
            for step in result['answer']['reasoning_path']:
                print(f"   Step {step['step']} - {step['type'].upper()}:")
                print(f"   └─ {step['description']}")

            # 5. 生成的答案
            print(f"\n{'=' * 80}")
            print(f"💡 GENERATED ANSWER")
            print(f"{'=' * 80}\n")
            print(f"Confidence Score: {result['answer']['confidence']:.3f}")
            print(f"\n{result['answer']['answer']}\n")

            # 6. 证据来源
            if result['answer']['sources']:
                num_sources = min(5, len(result['answer']['sources']))
                print(f"\n📖 Top {num_sources} Evidence Sources:\n")
                for i, source in enumerate(result['answer']['sources'][:num_sources], 1):
                    print(
                        f"   [{i}] Relevance: {source['quality_score']:.3f} | Similarity: {source['similarity']:.3f} | Hop: {source['hop']}")
                    # 截断长文本
                    text = source['text']
                    if len(text) > 200:
                        text = text[:200] + "..."
                    print(f"       \"{text}\"")
                    print()

            # 7. 元数据
            print(f"\n📊 Query Metadata:")
            print(f"   • Query Type: {result['analysis']['query_type']}")
            print(f"   • Evidence Integration: {result['evidence']['integration_method']}")
            print(f"   • Total Processing Steps: {len(result['answer']['reasoning_path'])}")

            print(f"\n{'=' * 80}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!\n")
            break
        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
            print(f"\n❌ Error: {e}")
            print("Please try another question.\n")

    # 显示会话统计
    if question_count > 0:
        print(f"\n{'=' * 80}")
        print(f"📈 Session Statistics:")
        print(f"   Total Questions Processed: {question_count}")
        print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description='Reasoning RAG System')
    parser.add_argument('--mode', type=str, default='demo',
                        choices=['build', 'eval', 'demo', 'interactive', 'all'],
                        help='Run mode: build (build index), eval (evaluation), demo (demo conversations), interactive (interactive Q&A), all (eval+demo)')
    parser.add_argument('--index-path', type=str, default='./bioasq_index.pkl',
                        help='Path to save/load the vector index')
    parser.add_argument('--eval-size', type=int, default=50,
                        help='Number of samples for evaluation')
    parser.add_argument('--demo-size', type=int, default=3,
                        help='Number of samples for demo')
    parser.add_argument('--rebuild-index', action='store_true',
                        help='Force rebuild the index even if it exists')
    parser.add_argument('--max-passages', type=int, default=1000,
                        help='Maximum number of passages for building index')

    args = parser.parse_args()

    # 初始化配置
    config = Config()

    # 初始化数据加载器
    data_loader = DataLoader()

    # 初始化RAG系统
    rag_system = ReasoningRAG(config)

    # 构建或加载索引
    if args.mode == 'build' or args.rebuild_index or not os.path.exists(args.index_path):
        build_index(rag_system, data_loader, args.index_path, args.max_passages)
    else:
        logger.info(f"Loading existing index from {args.index_path}")
        try:
            rag_system.load_index(args.index_path)
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            logger.info("Building new index...")
            build_index(rag_system, data_loader, args.index_path, args.max_passages)

        # 对于非build模式，需要加载数据集
        if args.mode != 'interactive':
            data_loader.load_bioasq_dataset(train_ratio=0.8)

    # 根据模式运行
    if args.mode == 'eval' or args.mode == 'all':
        metrics, results = run_evaluation(rag_system, data_loader, args.eval_size)

    if args.mode == 'demo' or args.mode == 'all':
        run_demo(rag_system, data_loader, args.demo_size)

    if args.mode == 'interactive':
        run_interactive(rag_system)

    logger.info("\n✅ All tasks completed!")


if __name__ == '__main__':
    main()