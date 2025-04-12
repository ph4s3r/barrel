# Project Barrel #

## Development environment ##
 - Add dependency: **`uv add <package-name>`**
 - Add dependency to a group **`uv sync --group <name-of-the-group> --all-groups`**
 - Update local dev dependencies: **`uv sync --all-groups`**

## Secrets ##
Prerequisites:
 - Virtual environment is created with `uv`

### Creating encrypted secrets ###
First, create an **.env** file based on **dot_env_template** if it does not exists, then set or update the credentials. To encrypt the secrets run the following command: `uv run python -c "from credentials.secrets import encrypt_env_file;encrypt_env_file()"`

## Testing ##
 - Execute all tests: **`uv run pytest`**
 - Execute subset of tests: **`uv run pytest -k "test_function_name"`**
