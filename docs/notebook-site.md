# Notebook site

The `notebooks/` directory is a Quarto website project. It renders checked-in
Jupyter notebooks to static HTML so they can be served by GitHub Pages instead
of relying on GitHub's repository notebook preview.

## Local setup

Install the Quarto CLI from <https://quarto.org/docs/get-started/>.

Render the site:

```bash
uv run --locked quarto render notebooks
```

Preview the site locally:

```bash
uv run --locked quarto preview notebooks
```

Rendered output is written to `notebooks/_site/`, which is ignored by Git.

Using `uv run` is optional for the current stored-output render, but it makes
Quarto see the repository `.venv` first if a future notebook or `.qmd` render
does need Python/Jupyter execution.

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
