from drnb.util import get_method_and_args

from .base import EmbeddingEval


def create_evaluators(
    eval_metrics: str | list[str] | None = None,
) -> list[EmbeddingEval]:
    """Create a list of embedding evaluators based on the given list of evaluation
    metrics. If eval_metrics is None, return an empty list.

    Possible evaluation metrics are:
    - "gs": Global score
    - "rte": Random triplet evaluation
    - "rpc": Random pair correlation evaluation
    - "nnp": Neighbourhood preservation evaluation
    - "unnp": Undirected neighbourhood preservation evaluation
    - "soccur": Second-order occurrence evaluation
    - "lp": Label preservation evaluation
    - "astress": Approximate stress evaluation
    - "exact-stress": Exact stress evaluation
    """

    if eval_metrics is None:
        return []

    if not isinstance(eval_metrics, (list, tuple)):
        eval_metrics = [eval_metrics]

    evaluators = []
    for embed_eval in eval_metrics:
        embed_eval_name, eval_kwds = get_method_and_args(embed_eval, {})

        embed_eval_name = embed_eval_name.lower()
        if embed_eval_name == "gs":
            from .globalscore import GlobalScore as ctor
        elif embed_eval_name == "rte":
            from .rte import RandomTripletEval as ctor
        elif embed_eval_name == "rpc":
            from .rpc import RandomPairCorrelEval as ctor
        elif embed_eval_name == "nnp":
            from .nbrpres import NbrPreservationEval as ctor
        elif embed_eval_name == "unnp":
            from .unbrpres import UndirectedNbrPreservationEval as ctor
        elif embed_eval_name == "soccur":
            from .soccur import SOccurrenceEval as ctor
        elif embed_eval_name == "lp":
            from .labelpres import LabelPreservationEval as ctor
        elif embed_eval_name == "astress":
            from .astress import ApproxStressEval as ctor
        elif embed_eval_name == "exact-stress":
            from .stress import StressEval as ctor
        else:
            raise ValueError(f"Unknown embed eval option '{embed_eval_name}'")
        evaluators.append(ctor(**eval_kwds))

    return evaluators
