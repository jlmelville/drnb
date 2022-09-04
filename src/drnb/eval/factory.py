from typing import Iterable

from .globalscore import GlobalScore
from .nbrpres import NbrPreservationEval
from .rpc import RandomPairCorrelEval
from .rte import RandomTripletEval


def create_evaluators(eval_metrics=None):
    if eval_metrics is None:
        return []

    if not isinstance(eval_metrics, Iterable):
        eval_metrics = [eval_metrics]

    evaluators = []
    for embed_eval in eval_metrics:
        if isinstance(embed_eval, tuple):
            if len(embed_eval) != 2:
                raise ValueError("Bad format for eval spec")
            embed_eval_name = embed_eval[0]
            eval_kwds = embed_eval[1]
        else:
            embed_eval_name = embed_eval
            eval_kwds = {}

        embed_eval_name = embed_eval_name.lower()
        if embed_eval_name == "gs":
            ctor = GlobalScore
        elif embed_eval_name == "rte":
            ctor = RandomTripletEval
        elif embed_eval_name == "rpc":
            ctor = RandomPairCorrelEval
        elif embed_eval_name == "nnp":
            ctor = NbrPreservationEval
        else:
            raise ValueError(f"Unknown embed eval option '{embed_eval_name}'")
        evaluators.append(ctor(**eval_kwds))

    return evaluators
