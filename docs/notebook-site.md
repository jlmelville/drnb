# Notebook site

The `notebooks/` directory is a Quarto website project. It renders selected
checked-in Jupyter notebooks to static HTML so they can be served by GitHub
Pages instead of relying on GitHub's repository notebook preview.

Only notebooks under `notebooks/articles/` are published. The rest of
`notebooks/` remains an archive of working notebooks and examples.

## Local setup

Install the Quarto CLI from <https://quarto.org/docs/get-started/>.

Render the site:

```bash
uv run --locked quarto render notebooks
```

Quarto has its own local preview server, which is useful while editing Quarto
configuration or article contents:

```bash
uv run --locked quarto preview notebooks
```

That server watches files and re-renders as needed. It does not run the
post-render PNG optimizer used by GitHub Pages.

For a preview that matches the GitHub Pages workflow more closely, run:

```bash
uv run --locked python scripts/preview_notebook_site.py
```

That script runs `quarto render notebooks`, optimizes rendered PNGs, and serves
`notebooks/_site/` with a local static HTTP server. It prints the preview URL,
using `http://127.0.0.1:4202/` by default or the next available port if that
port is already in use. Stop it with `Ctrl-C`.

To keep the static preview server running and rebuild when site inputs change:

```bash
uv run --locked python scripts/preview_notebook_site.py --watch
```

Watch mode rebuilds when files such as `notebooks/_quarto.yml`,
`notebooks/index.qmd`, `notebooks/_open-image.html`, or notebooks under
`notebooks/articles/` change. Refresh the browser after a rebuild completes.

To serve an already-rendered site without re-rendering:

```bash
uv run --locked python scripts/preview_notebook_site.py --no-render --no-optimize
```

Rendered output is written to `notebooks/_site/`, which is ignored by Git.

Using `uv run` is optional for the current stored-output render, but it makes
Quarto see the repository `.venv` first if a future notebook or `.qmd` render
does need Python/Jupyter execution.

Optimize rendered PNG images before publishing:

```bash
uv run --with pillow==11.1.0 --no-project python scripts/optimize_site_images.py notebooks/_site --max-edge 3200
```

The optimizer only touches rendered site images. It keeps notebook output images
unchanged, strips PNG metadata, recompresses PNGs, and downsamples images whose
width or height is larger than `--max-edge`.

The GitHub Pages workflow runs this optimizer after Quarto renders the site and
before uploading the Pages artifact.

## Rendered HTML tweaks

Quarto includes `notebooks/_open-image.html` after the body of each rendered
HTML page. That include adds small client-side behavior for stored notebook
outputs:

- figure images open in a new tab at their published image size when clicked;
- Rich-style logging output is collapsed into "Log output" disclosure blocks.
- on desktop, the sidebar is wider, long sidebar titles wrap, and the sidebar
  can be resized by dragging its right edge.

## Publishing an article

Move or copy the notebook into `notebooks/articles/`.

Private helper notebooks can also live in `notebooks/articles/`, but their
filenames must start with `_`, such as `_tfidf-renorm-prep.ipynb`. Those
notebooks are excluded from Quarto rendering and from the sidebar consistency
check.

Then add a sidebar entry to `notebooks/_quarto.yml`:

```yaml
website:
  sidebar:
    contents:
      - index.qmd
      - section: "Articles"
        contents:
          - text: my-article
            href: articles/my-article.ipynb
```

Using explicit `text` keeps sidebar labels based on notebook filenames rather
than Quarto's inferred titles or first markdown headings.

## Execution model

The site is configured with:

```yaml
execute:
  enabled: false
```

That keeps site builds lightweight and uses the outputs already saved in each
`.ipynb` file. If a notebook's plots or tables are stale, rerun that notebook in
Jupyter first, save it, and then render the Quarto site.

The current HTML site does not require a global Jupyter installation, TinyTeX,
or Chrome Headless Shell. Those are only needed for other workflows:

- Jupyter execution: rendering `.qmd` Python chunks or running notebooks with
  `--execute`.
- TinyTeX/LaTeX: PDF output.
- Chrome Headless Shell: web screenshots, browser-backed PDF output, or other
  browser-dependent formats.

## GitHub Pages

The `notebooks-site` workflow renders the site and deploys `notebooks/_site/`
as a GitHub Pages artifact.

Repository setup:

1. Go to repository Settings -> Pages.
2. Set Source to GitHub Actions.
3. Run the `Notebook site` workflow manually, or push a change under
   `notebooks/`.

The expected project URL is:

```text
https://jlmelville.github.io/drnb/
```
