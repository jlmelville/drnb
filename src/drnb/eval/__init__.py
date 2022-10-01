from dataclasses import dataclass, field

from drnb.embed import get_coords
from drnb.log import log
from drnb.util import Jsonizable, islisty


def evaluate_embedding(evaluators, X, embedding, ctx=None):
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


@dataclass
class EvalResult(Jsonizable):
    eval_type: str
    label: str
    value: float
    info: dict = field(default_factory=dict)
