from __future__ import annotations

"""
Query decomposer that breaks complex questions into sub-queries.
"""

from dataclasses import dataclass
from typing import List, Optional
import os

try:
    from openai import OpenAI
except ImportError as exc:
    raise ImportError("openai is required. Install with `pip install openai`.") from exc


@dataclass
class SubQuery:
    """A sub-query decomposed from the main question."""
    query_id: str
    text: str
    dependencies: List[str]  # IDs of sub-queries this depends on
    reasoning: str  # Why this sub-query is needed


class QueryDecomposer:
    """
    Decomposes complex questions into logical sub-queries.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def decompose(self, query: str, max_subqueries: int = 5) -> List[SubQuery]:
        """
        Break down a complex query into simpler sub-queries.
        """
        prompt = f"""Break down the following complex question into simpler sub-questions that can be answered independently.

Main Question: {query}

Provide up to {max_subqueries} sub-questions in this exact format:

SUB_QUERY_1:
TEXT: [the sub-question]
DEPENDENCIES: [comma-separated list of sub-query IDs this depends on, or "none"]
REASONING: [why this sub-question is needed]

SUB_QUERY_2:
...

Guidelines:
- Each sub-question should be self-contained and retrievable
- Order sub-questions logically (independent ones first)
- Dependencies use format: SUB_QUERY_1, SUB_QUERY_2, etc.
- Keep sub-questions focused and specific"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert at breaking down complex questions into simpler sub-questions."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        content = response.choices[0].message.content.strip()
        return self._parse_subqueries(content)

    def _parse_subqueries(self, content: str) -> List[SubQuery]:
        """Parse LLM response into SubQuery objects."""
        subqueries: List[SubQuery] = []
        current_id: Optional[str] = None
        current_data: dict = {}

        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Check for new sub-query
            if line.startswith('SUB_QUERY_'):
                # Save previous sub-query if exists
                if current_id and current_data:
                    subqueries.append(self._create_subquery(current_id, current_data))

                current_id = line.rstrip(':')
                current_data = {}
            elif ':' in line and current_id:
                key, value = line.split(':', 1)
                current_data[key.strip()] = value.strip()

        # Don't forget the last sub-query
        if current_id and current_data:
            subqueries.append(self._create_subquery(current_id, current_data))

        return subqueries

    def _create_subquery(self, query_id: str, data: dict) -> SubQuery:
        """Create SubQuery object from parsed data."""
        deps_str = data.get('DEPENDENCIES', 'none').lower()
        dependencies = []

        if deps_str != 'none':
            # Extract SUB_QUERY_X IDs
            deps = [d.strip() for d in deps_str.split(',')]
            dependencies = [d for d in deps if d.startswith('SUB_QUERY_')]

        return SubQuery(
            query_id=query_id,
            text=data.get('TEXT', ''),
            dependencies=dependencies,
            reasoning=data.get('REASONING', '')
        )