"""Unit tests."""
from pathlib import Path

import pytest
from cryptography.fernet import Fernet, InvalidToken

from credentials.secrets import get_private_key, decrypt_secrets, Secret


SECRET_FILE = Path(__file__).parent.parent / "credentials" / "private.key"


@pytest.mark.local_test_only
def test_get_private_key():
    """Test to load locally generated private keys."""
    assert SECRET_FILE.exists()
    assert get_private_key(SECRET_FILE)


@pytest.mark.local_test_only
def test_decrypt_secrets_with_invalid_secret_key():
    """Test decrypting an invalid key."""
    private_key = Fernet.generate_key()

    with pytest.raises(InvalidToken):
        decrypt_secrets(private_key)


@pytest.mark.local_test_only
def test_decrypt_secrets():
    """Test validity of managing credentials from environment variables."""
    assert SECRET_FILE.exists()
    private_key = get_private_key(SECRET_FILE)
    decrypt_secrets(private_key)

    secrets = Secret()

    for value in secrets.__dict__.values():
        assert value is not None
