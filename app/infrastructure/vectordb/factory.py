"""Factory for creating vector store instances based on configuration."""
from app.core.config import settings
from app.core.enums import VectorStoreType
from app.core.logging import get_logger
from app.domain.interfaces.storage.vector_store import VectorStore

logger = get_logger(__name__)


def create_vector_store() -> VectorStore:
    """Create a vector store instance based on configuration.
    
    Returns:
        VectorStore: The configured vector store implementation
        
    Raises:
        ValueError: If an unsupported vector store type is configured
    """
    store_type = settings.VECTOR_STORE_TYPE
    
    if store_type == VectorStoreType.PGVECTOR:
        from app.infrastructure.vectordb.pgvector import PGVectorStore
        logger.info("Creating PGVector vector store")
        return PGVectorStore()
    
    elif store_type == VectorStoreType.PINECONE:
        from app.infrastructure.vectordb.pinecone import PineconeVectorStore
        logger.info("Creating Pinecone vector store")
        return PineconeVectorStore()
    
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")
