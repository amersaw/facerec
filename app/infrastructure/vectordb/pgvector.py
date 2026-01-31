"""PGVector implementation of vector store for face embeddings."""

from typing import List, Optional
from datetime import datetime
import uuid

import numpy as np
import asyncpg
from pgvector.asyncpg import register_vector

from app.core.config import settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.domain.entities.face import BoundingBox, Face
from app.domain.interfaces.storage.vector_store import VectorStore
from app.domain.value_objects.recognition import FaceMatch, SearchResult

logger = get_logger(__name__)

# Embedding dimension for InsightFace buffalo_l model
EMBEDDING_DIM = 512


def transform_pgvector_cosine_distance(distance: float) -> float:
    """Transform PGVector cosine distance to [0, 100] similarity scale.

    PGVector's <=> operator returns cosine distance (1 - cosine_similarity),
    ranging from 0 (identical) to 2 (opposite).

    Args:
        distance: Cosine distance from PGVector (0 to 2)

    Returns:
        Transformed similarity score on 0-100 scale
    """
    # cosine_distance = 1 - cosine_similarity
    # So: cosine_similarity = 1 - distance (ranges from -1 to 1)
    similarity = 1 - distance
    # Transform from [-1, 1] to [0, 1]
    normalized = (similarity + 1) / 2
    # Scale to [0, 100]
    return normalized * 100


class PGVectorStore(VectorStore):
    """PGVector (PostgreSQL) implementation of vector store for face embeddings.

    This implementation stores face embeddings in PostgreSQL using the pgvector
    extension for efficient similarity search using HNSW or IVFFlat indexes.
    """

    def __init__(self) -> None:
        """Initialize PGVector connection pool.

        Raises:
            VectorStoreError: If initialization fails
        """
        self._pool: Optional[asyncpg.Pool] = None
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure the connection pool is initialized and table exists."""
        if self._initialized:
            return

        try:
            # Build connection string
            dsn = (
                f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
                f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
            )

            # Create connection pool
            self._pool = await asyncpg.create_pool(
                dsn, min_size=2, max_size=10, init=self._init_connection
            )

            # Ensure table and indexes exist
            await self._ensure_schema()

            self._initialized = True
            logger.info(
                "PGVector vector store initialized",
                host=settings.POSTGRES_HOST,
                database=settings.POSTGRES_DB,
            )

        except Exception as e:
            logger.error("Failed to initialize PGVector", error=str(e), exc_info=True)
            raise VectorStoreError(f"Failed to initialize PGVector: {str(e)}")

    async def _init_connection(self, conn: asyncpg.Connection) -> None:
        """Initialize a connection with pgvector extension."""
        await register_vector(conn)

    async def _ensure_schema(self) -> None:
        """Ensure the face_embeddings table and indexes exist."""
        async with self._pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create table
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id UUID PRIMARY KEY,
                    collection_id VARCHAR(255) NOT NULL,
                    image_key VARCHAR(1024) NOT NULL,
                    detection_id VARCHAR(255) NOT NULL,
                    confidence FLOAT NOT NULL,
                    bbox_left FLOAT NOT NULL,
                    bbox_top FLOAT NOT NULL,
                    bbox_width FLOAT NOT NULL,
                    bbox_height FLOAT NOT NULL,
                    embedding VECTOR({EMBEDDING_DIM}) NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """
            )

            # Create indexes (if not exists is implicit in CREATE INDEX IF NOT EXISTS)
            # Index for collection + id uniqueness
            await conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_face_embeddings_collection_id 
                ON face_embeddings (collection_id, id)
            """
            )

            # Index for image_key lookups within a collection
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_face_embeddings_collection_image 
                ON face_embeddings (collection_id, image_key)
            """
            )

            # HNSW index for fast cosine similarity search
            # Note: This may take time on large datasets
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_face_embeddings_hnsw 
                ON face_embeddings 
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """
            )

            logger.debug("PGVector schema ensured")

    async def store_face(
        self,
        face: Face,
        collection_id: str,
        image_key: str,
        face_detection_id: str,
        detection_id: str = None,
    ) -> None:
        """Store a face embedding in PostgreSQL.

        Args:
            face: Face object with embedding vector
            collection_id: Collection identifier for namespace isolation
            image_key: S3 object key of the original image
            face_detection_id: External system face detection identifier
            detection_id: ID grouping faces from same detection operation (optional)

        Raises:
            VectorStoreError: If storage operation fails
        """
        await self._ensure_initialized()

        try:
            if face.embedding is None:
                raise VectorStoreError(
                    "Face embedding is None, cannot store in vector database"
                )

            # Use face_detection_id as the primary ID
            vector_id = uuid.UUID(face_detection_id)

            # Use provided detection_id or fallback to face_detection_id
            group_detection_id = detection_id if detection_id else face_detection_id

            # Normalize the embedding for optimal cosine similarity
            embedding = face.embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                normalized_embedding = embedding / norm
            else:
                normalized_embedding = embedding

            now = datetime.now()

            async with self._pool.acquire() as conn:
                # Upsert the face embedding
                await conn.execute(
                    """
                    INSERT INTO face_embeddings (
                        id, collection_id, image_key, detection_id,
                        confidence, bbox_left, bbox_top, bbox_width, bbox_height,
                        embedding, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (collection_id, id) DO UPDATE SET
                        image_key = EXCLUDED.image_key,
                        detection_id = EXCLUDED.detection_id,
                        confidence = EXCLUDED.confidence,
                        bbox_left = EXCLUDED.bbox_left,
                        bbox_top = EXCLUDED.bbox_top,
                        bbox_width = EXCLUDED.bbox_width,
                        bbox_height = EXCLUDED.bbox_height,
                        embedding = EXCLUDED.embedding,
                        created_at = EXCLUDED.created_at
                """,
                    vector_id,
                    collection_id,
                    image_key,
                    group_detection_id,
                    face.confidence,
                    face.bounding_box.left,
                    face.bounding_box.top,
                    face.bounding_box.width,
                    face.bounding_box.height,
                    normalized_embedding.tolist(),
                    now,
                )

            logger.debug(
                "Stored face embedding",
                face_id=face_detection_id,
                detection_id=group_detection_id,
                collection_id=collection_id,
                image_key=image_key,
                created_at=now.isoformat(),
            )

        except VectorStoreError:
            raise
        except Exception as e:
            logger.error(
                "Failed to store face embedding",
                error=str(e),
                face_id=face_detection_id,
                collection_id=collection_id,
                image_key=image_key,
                exc_info=True,
            )
            raise VectorStoreError(f"Failed to store face embedding: {str(e)}")

    async def get_faces_by_image_key(
        self,
        image_key: str,
        collection_id: str,
    ) -> tuple[List[Face], Optional[str]]:
        """Retrieve face entities for a given image key from PostgreSQL.

        Args:
            image_key: S3 object key of the original image
            collection_id: Collection identifier

        Returns:
            Tuple of (list of Face entities, detection_id)

        Raises:
            VectorStoreError: If retrieval operation fails
        """
        await self._ensure_initialized()

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, detection_id, confidence, 
                           bbox_left, bbox_top, bbox_width, bbox_height,
                           embedding, created_at
                    FROM face_embeddings
                    WHERE collection_id = $1 AND image_key = $2
                    LIMIT 100
                """,
                    collection_id,
                    image_key,
                )

            if not rows:
                logger.debug(
                    "No faces found for image key",
                    image_key=image_key,
                    collection_id=collection_id,
                )
                return [], None

            # Get detection_id from first row
            detection_id = rows[0]["detection_id"] if rows else None

            if not detection_id:
                logger.warning(
                    "No detection_id found for faces",
                    image_key=image_key,
                    collection_id=collection_id,
                )
                return [], None

            faces = []
            for row in rows:
                try:
                    bbox = BoundingBox(
                        left=float(row["bbox_left"]),
                        top=float(row["bbox_top"]),
                        width=float(row["bbox_width"]),
                        height=float(row["bbox_height"]),
                    )

                    # Convert embedding back to numpy array
                    embedding = (
                        np.array(row["embedding"])
                        if row["embedding"] is not None
                        else None
                    )

                    face_entity = Face(
                        face_id=str(row["id"]),
                        confidence=float(row["confidence"]),
                        bounding_box=bbox,
                        embedding=embedding,
                        created_at=row["created_at"],
                    )
                    faces.append(face_entity)

                except Exception as e:
                    logger.error(
                        "Failed to parse stored face record into Face entity",
                        error=str(e),
                        face_id=str(row["id"]),
                        image_key=image_key,
                        collection_id=collection_id,
                        exc_info=True,
                    )
                    continue

            return faces, detection_id

        except Exception as e:
            logger.error(
                "Failed to retrieve faces by image key",
                error=str(e),
                image_key=image_key,
                collection_id=collection_id,
                exc_info=True,
            )
            raise VectorStoreError(f"Failed to retrieve faces by image key: {str(e)}")

    async def search_faces(
        self,
        query_face: Face,
        collection_id: str,
        similarity_threshold: Optional[float] = None,
        max_matches: Optional[int] = None,
    ) -> SearchResult:
        """Search for similar faces in PostgreSQL using pgvector.

        Args:
            query_face: Face to search for
            collection_id: Collection identifier
            similarity_threshold: Minimum similarity score (0-100)
            max_matches: Maximum number of matches to return

        Returns:
            SearchResult containing the matches found

        Raises:
            VectorStoreError: If search operation fails
        """
        await self._ensure_initialized()

        try:
            if query_face.embedding is None:
                raise VectorStoreError("Query face embedding cannot be None")

            # Generate a unique ID for this search operation
            search_operation_id = str(uuid.uuid4())
            logger.debug(f"Starting face search op {search_operation_id}")

            # Normalize the query embedding for cosine similarity
            query_embedding = query_face.embedding
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                normalized_embedding = query_embedding / norm
            else:
                normalized_embedding = query_embedding

            # Default max_matches to 100 if not specified
            top_k = max_matches if max_matches is not None else 100

            # Convert similarity threshold from [0, 100] to cosine distance
            # similarity = 100 means distance = 0, similarity = 0 means distance = 2
            if similarity_threshold is not None:
                # Transform from [0, 100] to [0, 1]
                normalized_threshold = similarity_threshold / 100
                # Transform from [0, 1] to [-1, 1] (cosine similarity)
                cosine_sim_threshold = (normalized_threshold * 2) - 1
                # Convert to distance: distance = 1 - similarity
                max_distance = 1 - cosine_sim_threshold
            else:
                max_distance = 2.0  # Maximum possible distance

            async with self._pool.acquire() as conn:
                # Query using cosine distance operator <=>
                rows = await conn.fetch(
                    """
                    SELECT id, image_key, embedding <=> $1::vector AS distance
                    FROM face_embeddings
                    WHERE collection_id = $2
                      -- AND embedding <=> $1::vector <= $3
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                """,
                    normalized_embedding.tolist(),
                    collection_id,
                    top_k,
                )

            if not rows:
                logger.debug(
                    "No matches found",
                    collection_id=collection_id,
                    threshold=similarity_threshold,
                )
                return SearchResult(searched_face_id="", face_matches=[])

            # Convert matches to FaceMatch objects
            face_matches = []
            for row in rows:
                try:
                    # Transform cosine distance to [0, 100] similarity scale
                    similarity = transform_pgvector_cosine_distance(row["distance"])

                    # Double-check threshold (should already be filtered by query)
                    if (
                        similarity_threshold is not None
                        and similarity < similarity_threshold
                    ):
                        continue

                    image_key_from_db = row["image_key"]
                    if image_key_from_db is None:
                        logger.warning(
                            "Missing image_key for match, skipping",
                            face_id=str(row["id"]),
                            collection_id=collection_id,
                        )
                        continue

                    face_matches.append(
                        FaceMatch(
                            face_id=str(row["id"]),
                            similarity=similarity / 100,
                            image_key=image_key_from_db,
                        )
                    )

                except Exception as e:
                    logger.error(
                        "Failed to parse face match",
                        error=str(e),
                        face_id=str(row["id"]),
                        collection_id=collection_id,
                        exc_info=True,
                    )
                    continue

            return SearchResult(
                searched_face_id=search_operation_id, face_matches=face_matches
            )

        except VectorStoreError:
            raise
        except Exception as e:
            logger.error(
                "Failed to search faces",
                error=str(e),
                collection_id=collection_id,
                threshold=similarity_threshold,
                exc_info=True,
            )
            raise VectorStoreError(f"Failed to search faces: {str(e)}")

    async def delete_face(
        self,
        face_detection_id: str,
        collection_id: str,
    ) -> None:
        """Delete a face embedding from PostgreSQL.

        Args:
            face_detection_id: External system face detection identifier
            collection_id: Collection identifier

        Raises:
            VectorStoreError: If deletion operation fails
        """
        await self._ensure_initialized()

        try:
            vector_id = uuid.UUID(face_detection_id)

            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    DELETE FROM face_embeddings
                    WHERE collection_id = $1 AND id = $2
                """,
                    collection_id,
                    vector_id,
                )

            logger.info(
                "Deleted face",
                face_detection_id=face_detection_id,
                collection_id=collection_id,
            )

        except Exception as e:
            logger.error(
                "Failed to delete face",
                error=str(e),
                face_detection_id=face_detection_id,
                collection_id=collection_id,
                exc_info=True,
            )
            raise VectorStoreError(f"Failed to delete face: {str(e)}")

    async def delete_collection(
        self,
        collection_id: str,
    ) -> None:
        """Delete all face embeddings in a collection from PostgreSQL.

        Args:
            collection_id: Collection identifier

        Raises:
            VectorStoreError: If deletion operation fails
        """
        await self._ensure_initialized()

        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(
                    """
                    DELETE FROM face_embeddings
                    WHERE collection_id = $1
                """,
                    collection_id,
                )

            logger.info(
                "Deleted collection", collection_id=collection_id, result=result
            )

        except Exception as e:
            logger.error(
                "Failed to delete collection",
                error=str(e),
                collection_id=collection_id,
                exc_info=True,
            )
            raise VectorStoreError(f"Failed to delete collection: {str(e)}")

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._initialized = False
            logger.info("PGVector connection pool closed")
