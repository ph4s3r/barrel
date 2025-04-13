import voyageai

from credentials.secrets import secrets


def get_embedder_client():
    """Initialize Voyage client."""
    return voyageai.Client(
        api_key=secrets.embedder_client_api_key,
        max_retries=2,
        # timeout=5.0  # Timeout in seconds.
    )
