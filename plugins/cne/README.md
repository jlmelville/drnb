# Contrastive Neighbor-Embedding

Like `PyMDE`, this plugin makes use of PyTorch. You may well want to change the version of PyTorch
in the `pyproject.toml` to get the most out of your GPU. As with PyMDE, I have selfishly chosen a
version that still works with my laptop 1080 GPU. To compound the misery, I am attempting to make
this work with WSL.

This library also makes use of PyKeOps to write and compile CUDA kernels. I got errors of the type
`/usr/bin/ld: cannot find -lnvrtc: No such file or directory` when running this. My solution was
to find where `libnvrtc.so` lived via `sudo find /usr -name libnvrtc.so*` and then soft linking
the file to `/usr/lib`. For my Ubuntu 25.10 release that involved
`sudo ln -s /usr/local/cuda-12.5/targets/x86_64-linux/lib/libnvrtc.so /usr/lib/libnvrtc.so`. I
assume there are better ways involving `LD_LIBRARY_PATH` though.
