"""配置文件"""

class Config:
    # 模型配置
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # 检索配置
    TOP_K_RETRIEVAL = 5
    SIMILARITY_THRESHOLD = 0.3

    # 查询分解配置
    MAX_SUBQUERIES = 4
    COMPLEXITY_THRESHOLD = 0.2  # 超过此阈值认为是复杂问题

    # 多跳检索配置
    MAX_HOPS = 3

    # 答案生成配置
    MAX_EVIDENCE_LENGTH = 2000

    # 评估配置
    TEST_SAMPLE_SIZE = 50
    DEMO_SAMPLE_SIZE = 3