from drnb.embed import get_coords


def evaluate_embedding(evaluators, X, embedding):
    coords = get_coords(embedding)
    return [evaluator.evaluate(X, coords) for evaluator in evaluators]
