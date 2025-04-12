"""Securely manage secrets."""
import os
from pathlib import Path
from io import StringIO

from dotenv import load_dotenv
from cryptography.fernet import Fernet


dot_env_file = Path(__file__).parent / ".env"
public_key_file = Path(__file__).parent / "public.key"


def encrypt_env_file() -> None:
    """Create encrypted public and private keys based on .env files."""
    private_key = Fernet.generate_key()
    cipher_suit = Fernet(private_key)

    with open(dot_env_file, "r", encoding="UTF-8") as file:
        content = file.read()

    encrypted_data = cipher_suit.encrypt(bytes(content, encoding="UTF-8"))

    # Save the public key into a file
    with open(public_key_file, "wb") as file:
        file.write(encrypted_data)

    # Save the private key into a file
    with open(dot_env_file.parent / "private.key", "wb") as file:
        file.write(private_key)


def get_private_key(file_path: Path) -> bytes:
    """Read private key file."""
    with open(file_path, "rb") as file:
        private_key = file.read()

    return private_key

def decrypt_secrets(private_key: str | bytes) -> None:
    """Decrypt the private key and load them as env variables."""
    cipher_suit = Fernet(private_key)

    with open(public_key_file, "r", encoding="UTF-8") as file:
        encrypted_data = file.read()

    decrypted_data = cipher_suit.decrypt(encrypted_data)

    if not load_dotenv(stream=StringIO(decrypted_data.decode())):
        raise EnvironmentError("Failed to create environment variables from decrypted keys.")


class Secret:
    """Credential manager class."""

    def __init__(self):
        """Load secrets from env variables."""
        self.embedder_client_api_key = os.environ["EMBEDDER_API_KEY"]
        self.vector_db_api_key = os.environ["VECTOR_DB_API_KEY"]
        # Evaluator LLM API key is optional
        self.test_eval_llm_api_key = os.getenv("TEST_EVALUATOR_LLM_API_KEY")


if dot_env_file.exists():
    load_dotenv(dotenv_path=dot_env_file)
elif os.getenv("DOTENV_SECRET_FILE"):
    decrypt_secrets(os.environ["DOTENV_SECRET_FILE"])
else:
    raise EnvironmentError("ERROR in loading project secrets")

# Import this variable directly from this file as a singleton
secrets = Secret()
