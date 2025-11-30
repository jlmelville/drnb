# FAISS

FAISS is used internally by drnb for **exact** nearest neighbors. This definitely needs to be on
the GPU to not be way too slow for most cases. However, installing the GPU-powered FAISS is not
very easy via `pip`.

If you just install this package the way it currently is, it is going to not do anything. So if
you want FAISS support you will need to do some work.

In general what I do is build from source, which is also not very easy, but I made my life even
harder by installing a version that will work with a laptop 1080 card and on WSL. You can see my
[notes on installing on WSL](https://gist.github.com/jlmelville/9b4f0d91ede13bff18d26759140709f9)
in case that helps anyone.

Assuming you have built from source ok, you should, starting from this folder:

```bash
# install if necessary and create the virtual environment for this plugin
uv sync
source .venv/bin/activate

# change this to the folder you built faiss in
cd ~/dev/faiss/build/faiss/python

python setup.py install
```

This will install faiss into the plugin.

I haven't tried to build from source in quite a while (newer versions may not even support the 
older CUDA version I need for the 1080), so some of this may change. For example, you may not need
to use `setuptools` any more, in which case you can remove that dependency from the `pyproject.toml`
here.

Your NVidia graphics card may be a lot newer than mine (it could hardly be older), so most of this
may be unnecessary, in which case edit the `pyproject.toml` file appropriately.
