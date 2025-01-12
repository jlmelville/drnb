import abc
from dataclasses import dataclass, field
from typing import List

import numpy as np

from drnb.embed import get_coords
from drnb.embed.context import EmbedContext
from drnb.log import log
from drnb.util import islisty


@dataclass
class EvalResult:
    """Result of an embedding evaluation.

    Attributes:
    eval_type: str - type of evaluation (e.g. "RTE")
    label: str - label for the evaluation (e.g. "RandomTripletEval")
    value: float - evaluation result
    info: dict - additional information about the evaluation
    """

    eval_type: str = ""
    label: str = ""
    value: float = 0.0
    info: dict = field(default_factory=dict)


class EmbeddingEval(abc.ABC):
    """Base class for embedding evaluators. Subclasses must implement the evaluate
    method and may optionally implement the requires method.

    The evaluate method should return an EvalResult object."""

    def requires(self) -> dict:
        """Return a dictionary of requirements for this evaluator. Used when exporting
        an embedding to ensure that all necessary data is present."""
        return {}

    def evaluate(
        self, X: np.ndarray, coords: np.ndarray, _: EmbedContext | None = None
    ) -> EvalResult:
        """Evaluate the embedding. Return an EvalResult object."""
        raise NotImplementedError


def evaluate_embedding(
    evaluators: List[EmbeddingEval],
    X: np.ndarray,
    embedding: tuple | dict | np.ndarray,
    ctx: EmbedContext | None = None,
) -> List[EvalResult]:
    """Evaluate an embedding using a list of EmbeddingEval evaluators. Return a list of
    EvalResult objects."""

    coords = get_coords(embedding)

    eval_results = []
    for evaluator in evaluators:
        log.info(evaluator)
        eval_result = evaluator.evaluate(X, coords, ctx=ctx)
        if islisty(eval_result):
            eval_results += eval_result
        else:
            eval_results.append(eval_result)
    return eval_results
