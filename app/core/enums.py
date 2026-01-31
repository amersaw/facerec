"""Application enums."""
from enum import Enum


class VectorStoreType(str, Enum):
    """Supported vector store backends."""
    PINECONE = "pinecone"
    PGVECTOR = "pgvector"
