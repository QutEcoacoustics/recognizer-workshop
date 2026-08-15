# Recognizer Workshop

A web app for building acoustic event recognizers using agile modelling, built on
[perch-hoplite](https://github.com/google-research/perch-hoplite) and the
[Bioacoustics Workbench](https://github.com/QutEcoacoustics/baw-server) (Ecosounds / Australian Acoustic Observatory).

The app guides you through an active-learning loop: embed audio into a searchable database,
find examples of a target call, label them, train a small classifier on top of the embeddings,
refine it with further labelling rounds, and run inference across the whole database.


## Installation

Requires **Python 3.11+** and ffmpeg (for audio decoding).
Recommend installing in a virtual environment.
Use whatever environment management you prefer (conda, venv, etc.).

```bash

# Clone including the perch-hoplite submodule
git clone --recurse-submodules https://github.com/peichins/recognizer-workshop.git
cd recognizer-workshop

# Install perch-hoplite from the submodule, then the two local packages (in this order)
pip install libs/perch-hoplite
pip install packages/baw-helpers
pip install packages/recognizer-workshop
```

### Installing without git

GitHub's "Download ZIP" does not include the perch-hoplite submodule, so if you don't have git,
install all three packages directly from archive URLs instead (perch-hoplite pinned to the same
commit as the submodule):

```bash
pip install https://github.com/google-research/perch-hoplite/archive/b075df9dfe2eedbaebfd3e97a52b4cffe3569fe2.zip
pip install "baw-helpers @ https://github.com/peichins/recognizer-workshop/archive/refs/heads/main.zip#subdirectory=packages/baw-helpers"
pip install "recognizer-workshop @ https://github.com/peichins/recognizer-workshop/archive/refs/heads/main.zip#subdirectory=packages/recognizer-workshop"
```

## Running the app

```bash
python -m recognizer_workshop.ui.panel.serve --port 5006 --address 127.0.0.1
```

Then open `http://127.0.0.1:5006` in a browser. Useful options:

- `--app-settings path/to/settings.json` — override default app settings
- `--autoreload` — reload on code changes during development

## Development

Install the two local packages editable instead, so code changes take effect without reinstalling:

```bash
pip install libs/perch-hoplite
pip install -e packages/baw-helpers
pip install -e packages/recognizer-workshop
```

Run the tests with:

```bash
PYTEST_TOKEN_POLICY=skip pytest packages/recognizer-workshop/tests
```

API tokens are needed for some integration tests, which talk to a live Workbench instance.
Provide a `STAGING_API_TOKEN` and a `STAGING_API_TOKEN` in a `.env` file at the repo root to run them; with
`PYTEST_TOKEN_POLICY=skip` they are skipped instead.

## License

Apache 2.0 — see [LICENSE](LICENSE).
