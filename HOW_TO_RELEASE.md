# How to issue a new aitana release

## Running tests
Run the tests before every release.

### All packages
From the repository root, run the tests for every package (`aitana-*/tests`) plus the cross-package integration tests in `tests/`:
```
uv run pytest
```
This uses the workspace environment, so all packages are installed from your local checkout.

### Hatch test matrix (per package)
`aitana-access`, `aitana-features`, `aitana-storage` and `aitana-volcanoes` define a hatch `test` matrix that runs the package's tests in isolated environments for Python 3.11, 3.12 and 3.13. Run it from inside the package directory:
```
cd aitana-storage
uvx hatch run test:run-pytest
```
To run a single Python version from the matrix:
```
uvx hatch run +py=3.12 test:run-pytest
```
Arguments after the script name are passed to pytest, e.g. `uvx hatch run test:run-pytest -x -k zarr`.

Hatch does not use the uv workspace, so sibling packages (e.g. `aitana-storage` as a dependency of `aitana-access`) are installed from PyPI. If you have changed `aitana-storage`, release it before running the `aitana-access` matrix. Likewise, `aitana-testing` (a dependency of `aitana-access` and of the `aitana-features` tests) must be released before those matrices can run.

### Optional tests
Slow tests and tests that request data from webservices are skipped by default. Enable them with:
```
uv run pytest --runslow --runwebservice
```
The same flags also work with the hatch commands above.

## Pypi
Install the build system
```
python3 -m pip install --upgrade build
```
Install `twine`:
```
python3 -m pip install --upgrade twine
```
Make sure the version number is incremented in the `pyproject.toml` file.

Run the build to create two new files under `dist`. One is the wheel and the other the packaged source code:
```
python3 -m build
```
Upload to pypi:
```
python3 -m twine upload dist/*
```
When prompted for the username enter `__token__` and paste in your pypi token as the password.

## Documentation
The documentation is built with [Great Docs](https://posit-dev.github.io/great-docs/) (part of the workspace dev dependencies), which needs [Quarto](https://quarto.org/docs/get-started/) (preinstalled in the devcontainer). It is configured in `great-docs.yml`; the homepage is `README.md`, and the pages come from `user_guide/` and `developer/`. The API reference lists every class and function explicitly in `great-docs.yml`, so add new public ones there.

To generate the documentation run:
```
uv run great-docs build
```
The user guide pages execute their code, so build from the synced workspace (`uv sync`). The site is written to `great-docs/_site`.

To view the documentation locally run:
```
uv run great-docs preview
```

To publish the documentation to [Github pages](https://pages.github.com/) run:

```
uvx ghp-import -n -p -r origin great-docs/_site
```
where origin is the remote name that you have given to the github repo. This will then publish the documentation under https://tsc-tools.github.io/aitana