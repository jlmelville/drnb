import abc


class EmbeddingEval(abc.ABC):
    def evaluate(self, X, coords):
        pass
