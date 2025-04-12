# Project Barrel #

## Development environment ##
Dependencies are managed by [uv](https://docs.astral.sh/uv/)
 - Create a `pyproject.toml` file for an existing project: **`uv init --bare`**
 - List all available / locally installed interpreters: **`uv python list`**
 - Install a specific Python interpreter: **`uv python install 3.12`**
 - Add dependency: **`uv add <package-name>`**
 - Add dependency to a custom dependency group **`uv add --group <name-of-the-group> <package-name>`**
 - Create / update local virtual environment: **`uv sync --all-groups`**
 - Display dependency tree: **`uv tree --all-groups`**
 - Run a script with activated virtual environment: **`uv run python <python-file>`**

## Secrets ##
Prerequisites:
 - Virtual environment is created with `uv`

### Creating encrypted secrets ###
First, create an **.env** file based on **dot_env_template** if it does not exists, then set or update the credentials. To encrypt the secrets run the following command: `uv run python -c "from credentials.secrets import encrypt_env_file;encrypt_env_file()"`

## Launch the backed service ##
 - root-repo-folder>: **`uv run uvicorn main:app --reload`**

## Testing ##
Tests are written to fit for [pytest](https://docs.pytest.org/en/stable/)
 - Execute all tests: **`uv run pytest`**
 - Execute subset of test(s): **`uv run pytest -k "test_function_name"`**
