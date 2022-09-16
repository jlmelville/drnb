For the notebooks here to work, you must set the `DRNB_HOME` environment variable to a root
directory where you want DRNB to work, including creating sub-folders. For example if you want
dataset files to live in `/home/you/drnb/data`, then set the home folder to the parent of that,
i.e.:

```bash
export DRNB_HOME=/home/you/drnb
```

And you need to do this *before* starting the JupyterLab server. So if you are reading this for the
first time inside JupyterLab, sorry.