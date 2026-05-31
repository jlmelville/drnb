# pymde

This uses PyTorch for optimization and there are lots of ways for this to go wrong if you want to
use a GPU (which you probably do).

## CUDA support

As CUDA support changes so will the version of the PyTorch binary you can download, and I just want
to try my luck with what is available with `uv add torch`. In my case,
I wanted my laptop 1080 to still work. The discussion here was helpful:
<https://discuss.pytorch.org/t/what-version-of-pytorch-is-compatible-with-nvidia-geforce-gtx-1080/222056>.
In this case it meant I wanted to use CUDA version 12.6, which in turn meant 2.6 was ok. Life can
get more complicated if you need a version that doesn't support Python 3.13 because you will then
be unable to use the current drnb plugin SDK.

The plugin currently uses `pymde` 0.3.0 but keeps `torch>=2.6,<2.7` for the local old-GPU profile.
You may be okay with a newer PyTorch build on newer hardware, but you may need to edit
`pyproject.toml` to change the `requires-python` and torch dependency version together.
Unfortunately, installation may look like it succeeds and then fail at runtime. MPS is also not yet
supported.
