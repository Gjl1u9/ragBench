import pytest
from qdrant_client import QdrantClient

from ragbench.config import settings


def test_settings_load():
    assert settings.qdrant_url.startswith("http")


@pytest.mark.integration
def test_qdrant_reachable():
    client = QdrantClient(url=settings.qdrant_url)
    client.get_collections()
