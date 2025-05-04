# How to run

git config user.name yourname
git config user.email youremail
git clone https://github.com/ph4s3r/barrel.git
cd barrel

# create a token if don't have one https://github.com/settings/personal-access-tokens

cp ./credentials/dot_env_template ./credentials/.env
### enter API keys
uv run python -c "from credentials.secrets import encrypt_env_file;encrypt_env_file()"
# hosting on 0.0.0.0 for external access (dev)
uv run uvicorn main:app --reload --host 0.0.0.0