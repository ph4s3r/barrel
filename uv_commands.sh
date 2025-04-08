# UV docs:
# https://docs.astral.sh/uv/getting-started/



# list currently installed interpreters
uv python list --only-installed
# list all available interpreters to download
uv python list





# init: creates a pyproject.toml file
uv init --bare

# create venv
uv venv create 3.12
# switch to venv
uv venv use 3.12



# install a specific python version
# https://docs.astral.sh/uv/guides/install-python/#installing-a-specific-version
uv python install 3.12

# pip install
uv add fastapi
uv add onnxruntime==1.15.0

# pip uninstall
uv remove fastapi

# just run a python file
uv run python .\embedding_client.py

# run the fastapi app
uv run uvicorn main:app --reload

# display package tree
uv tree