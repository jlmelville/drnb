# torchknn

A very brain-dead exact nearest neighbors module using PyTorch for GPU or MPS acceleration. It
won't be as fast as FAISS, but it has the advantage of being more easily installed (you just
need PyTorch), and it's still faster than using the CPU.

As with all plugins that use PyTorch, the version included here uses an older version of PyTorch
to work with my old GPU. You should update the `pyproject.toml` to use a more modern PyTorch if
you prefer and then run `uv sync --reinstall-package drnb-nn-plugin-torchknn`.
