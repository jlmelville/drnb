from typing import List

from drnb.util import get_method_and_args

from .astress import ApproxStressEval
from .base import EmbeddingEval
from .globalscore import GlobalScore
from .labelpres import LabelPreservationEval
from .nbrpres import NbrPreservationEval
from .rpc import RandomPairCorrelEval
from .rte import RandomTripletEval
from .soccur import SOccurrenceEval
from .stress import StressEval
from .unbrpres import UndirectedNbrPreservationEval


def create_evaluators(
    eval_metrics: str | List[str] | None = None,
) -> List[EmbeddingEval]:
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
            ctor = GlobalScore
        elif embed_eval_name == "rte":
            ctor = RandomTripletEval
        elif embed_eval_name == "rpc":
            ctor = RandomPairCorrelEval
        elif embed_eval_name == "nnp":
            ctor = NbrPreservationEval
        elif embed_eval_name == "unnp":
            ctor = UndirectedNbrPreservationEval
        elif embed_eval_name == "soccur":
            ctor = SOccurrenceEval
        elif embed_eval_name == "lp":
            ctor = LabelPreservationEval
        elif embed_eval_name == "astress":
            ctor = ApproxStressEval
        elif embed_eval_name == "exact-stress":
            ctor = StressEval
        else:
            raise ValueError(f"Unknown embed eval option '{embed_eval_name}'")
        evaluators.append(ctor(**eval_kwds))

    return evaluators
