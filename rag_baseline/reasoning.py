from __future__ import annotations

"""
Question complexity analyzer and strategy selector.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import os

try:
    from openai import OpenAI
except ImportError as exc:
    raise ImportError("openai is required. Install with `pip install openai`.") from exc


class QuestionComplexity(Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"


@dataclass
class QuestionAnalysis:
    """Analysis result of a user question."""
    complexity: QuestionComplexity
    reasoning_type: str
    needs_decomposition: bool
    explanation: str


class QuestionAnalyzer:
    """
    Analyzes user questions to determine if they need multi-hop reasoning.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def analyze(self, query: str) -> QuestionAnalysis:
        """
        Analyze question complexity and determine if decomposition is needed.
        """
        prompt = f"""Analyze the following question and determine its complexity:

Question: {query}

Respond in this exact format:
COMPLEXITY: [simple/complex]
REASONING_TYPE: [factual/comparative/multi-step/causal/other]
NEEDS_DECOMPOSITION: [yes/no]
EXPLANATION: [brief explanation]

Guidelines:
- SIMPLE: Can be answered with a single retrieval step
- COMPLEX: Requires multiple pieces of information, comparison, or multi-step reasoning
- Answer concisely"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert at analyzing question complexity."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )

        content = response.choices[0].message.content.strip()
        return self._parse_analysis(content)

    def _parse_analysis(self, content: str) -> QuestionAnalysis:
        """Parse LLM response into QuestionAnalysis object."""
        lines = content.split('\n')
        parsed = {}

        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                parsed[key.strip()] = value.strip()

        complexity_str = parsed.get('COMPLEXITY', 'simple').lower()
        complexity = QuestionComplexity.COMPLEX if 'complex' in complexity_str else QuestionComplexity.SIMPLE

        needs_decomp_str = parsed.get('NEEDS_DECOMPOSITION', 'no').lower()
        needs_decomposition = 'yes' in needs_decomp_str

        return QuestionAnalysis(
            complexity=complexity,
            reasoning_type=parsed.get('REASONING_TYPE', 'factual'),
            needs_decomposition=needs_decomposition,
            explanation=parsed.get('EXPLANATION', 'No explanation provided')
        )