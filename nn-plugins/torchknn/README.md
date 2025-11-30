# torchknn

A very brain-dead exact nearest neighbors module using PyTorch for GPU or MPS acceleration. It
won't be as fast as FAISS, but it has the advantage of being more easily installed (you just
need PyTorch), and it's still faster than using the CPU.

As with all plugins that use PyTorch, the version included here uses an older version of PyTorch
to work with my old GPU. You should update the `pyproject.toml` to use a more modern PyTorch if
you prefer and then run `uv sync --reinstall-package drnb-nn-plugin-torchknn`.

## Supported metrics

* `euclidean` (default): exact L2 distance.
* `cosine`: rows are L2-normalized on device, neighbors are selected by highest cosine similarity, and distances returned are `1 - cosine_similarity` (so lower is closer, same ordering as other distances).

## Parameters

The one parameter you may need to change is `batch_size`. I don't know of a good way to estimate
an appropriate batch size easily, so it defaults to `1024`. Modify that if that seems like it will
be too small (or worse too large for your VRAM).
