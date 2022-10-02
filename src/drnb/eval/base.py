import abc


class EmbeddingEval(abc.ABC):
    def requires(self):
        return {}

    def evaluate(self, X, coords, ctx=None):
        pass
