from drnb.embed import get_coords
from drnb.log import log


def evaluate_embedding(evaluators, X, embedding):
    coords = get_coords(embedding)

    eval_results = []
    for evaluator in evaluators:
        log.info(evaluator)
        eval_results.append(evaluator.evaluate(X, coords))
    return eval_results
