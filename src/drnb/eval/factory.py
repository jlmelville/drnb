from typing import Iterable

from drnb.util import get_method_and_args

from .astress import ApproxStressEval
from .globalscore import GlobalScore
from .nbrpres import NbrPreservationEval
from .rpc import RandomPairCorrelEval
from .rte import RandomTripletEval
from .stress import StressEval


def create_evaluators(eval_metrics=None):
    if eval_metrics is None:
        return []

    if not isinstance(eval_metrics, Iterable):
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
        elif embed_eval_name == "astress":
            ctor = ApproxStressEval
        elif embed_eval_name == "exact-stress":
            ctor = StressEval
        else:
            raise ValueError(f"Unknown embed eval option '{embed_eval_name}'")
        evaluators.append(ctor(**eval_kwds))

    return evaluators
