# pymde

This uses PyTorch for optimization and there are lots of ways for this to go wrong if you want to
use a GPU (which you probably do).

## CUDA support

As CUDA support changes so will the version of the PyTorch binary you can download, and I just want
to try my luck with what is available with `uv add torch`. In my case,
I wanted my laptop 1080 to still work. The discussion here was helpful:
<https://discuss.pytorch.org/t/what-version-of-pytorch-is-compatible-with-nvidia-geforce-gtx-1080/222056>.
In this case it meant I wanted to use CUDA version 12.6, which in turn meant 2.6 was ok. Life can
get more complicated if you need a version that doesn't support python 3.12 because you will then
be unable to use the drnb plugin-sdk.

As of November 2025 `pymde` (at version 0.2.3) supports any torch >= 1.7.1, so you are probably ok
with a lot of GPUs although who knows how long you will be able to download PyTorch 2.6. But you may
need to edit the `pyproject.toml` to edit the `requires-python` and torch dependency version.
Unfortunately, installation may look like it succeeds and then fail at runtime. MPS is also not yet
supported.
