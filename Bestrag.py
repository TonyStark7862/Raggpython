"""EnhancedBestRAG"""
# Authors: Based on work by Abdul Samad Siddiqui <abdulsamadsid1@gmail.com>

import re
import uuid
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance
from fastembed import TextEmbedding
from fastembed.sparse.bm25 import Bm25
import PyPDF2


class EnhancedBestRAG:
    """
    EnhancedBestRAG is an improved version of BestRAG that uses local Qdrant persistence
    and adds reranking functionality for better search results.

    It supports dense embeddings, sparse embeddings, and a sophisticated reranking
    mechanism based on late interaction models for improved retrieval performance.

    Args:
        collection_name (str): The name of the Qdrant collection to use.
        persistence_path (str, optional): Path for local Qdrant storage. Defaults to "./qdrant_data".
        dense_model_name (str, optional): Model name for dense embeddings. Defaults to "all-MiniLM-L6-v2".
        late_interaction_model_name (str, optional): Model for reranking. Defaults to "colbertv2.0".
    """

    def __init__(self,
                 collection_name: str,
                 persistence_path: str = "./qdrant_data",
                 dense_model_name: str = "all-MiniLM-L6-v2",
                 late_interaction_model_name: str = "colbertv2.0"
                 ):
        self.collection_name = collection_name
        self.persistence_path = persistence_path
        
        # Initialize local Qdrant client
        self.client = QdrantClient(path=persistence_path)

        # Initialize embedding models
        self.dense_model = TextEmbedding(dense_model_name)
        self.late_interaction_model = TextEmbedding(late_interaction_model_name)
        self.sparse_model = Bm25("Qdrant/bm25")

        self._create_or_use_collection()

    def _create_or_use_collection(self):
        """
        Create a new Qdrant collection if it doesn't exist, or use an existing one.
        """
        collections = self.client.get_collections()
        collection_names = [
            collection.name for collection in collections.collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense-vector": models.VectorParams(
                        size=384,
                        distance=Distance.COSINE
                    ),
                    "output-token-embeddings": models.VectorParams(
                        size=384,
                        distance=Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        )
                    ),
                },
                sparse_vectors_config={"sparse": models.SparseVectorParams()}
            )
            print(f"Created collection: {self.collection_name}")
        else:
            print(f"Using existing collection: {self.collection_name}")

    def _clean_text(self, text: str) -> str:
        """
        Pass through the text without cleaning.
        Text cleaning has been removed as requested.

        Args:
            text (str): The text to be returned unchanged.

        Returns:
            str: The original text.
        """
        return text

    def _get_dense_embedding(self, text: str):
        """
        Get the dense embedding for the given text.

        Args:
            text (str): The text to be embedded.

        Returns:
            List[float]: The dense embedding vector.
        """
        return list(self.dense_model.embed([text]))[0]

    def _get_late_interaction_embedding(self, text: str):
        """
        Get the late interaction embedding for the given text.

        Args:
            text (str): The text to be embedded.

        Returns:
            List[float]: The late interaction embedding vector.
        """
        return list(self.late_interaction_model.embed([text]))[0]

    def _get_sparse_embedding(self, text: str):
        """
        Get the sparse embedding for the given text.

        Args:
            text (str): The text to be embedded.

        Returns:
            models.SparseVector: The sparse embedding vector.
        """
        return next(self.sparse_model.embed(text))

    def _extract_pdf_text_per_page(self, pdf_path: str) -> List[str]:
        """
        Load a PDF file and extract the text from each page.

        Args:
            pdf_path (str): The path to the PDF file.

        Returns:
            List[str]: The text from each page of the PDF.
        """
        with open(pdf_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            return [page.extract_text() for page in reader.pages]

    def store_pdf_embeddings(self, pdf_path: str,
                             pdf_name: str,
                             metadata: Optional[dict] = None):
        """
        Store the embeddings for each page of a PDF file in the Qdrant collection.

        Args:
            pdf_path (str): The path to the PDF file.
            pdf_name (str): The name of the PDF file.
            metadata (Optional[dict]): Additional metadata to store with each embedding.
        """
        texts = self._extract_pdf_text_per_page(pdf_path)

        for page_num, text in enumerate(texts):
            clean_text = self._clean_text(text)
            dense_embedding = self._get_dense_embedding(clean_text)
            late_interaction_embedding = self._get_late_interaction_embedding(
                clean_text)
            sparse_embedding = self._get_sparse_embedding(clean_text)

            hybrid_vector = {
                "dense-vector": dense_embedding,
                "output-token-embeddings": late_interaction_embedding,
                "sparse": models.SparseVector(
                    indices=sparse_embedding.indices,
                    values=sparse_embedding.values,
                )
            }

            # Store both content and all metadata in the payload
            # This ensures everything is indexed and searchable
            payload = {
                "text": clean_text,
                "page_number": page_num + 1,
                "pdf_name": pdf_name,
                "metadata": {}  # Create a nested metadata field
            }

            # Add any additional metadata
            if metadata:
                # Store metadata both at top level (for direct filtering)
                # and in nested field (for organization)
                payload.update(metadata)
                payload["metadata"] = metadata.copy()

            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=hybrid_vector,
                payload=payload
            )

            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            print(f"Stored embedding for page {page_num + 1} of '{pdf_name}' in collection '{self.collection_name}'.")

    def delete_pdf_embeddings(self, pdf_name: str):
        """
        Delete all embeddings associated with a given PDF name from the Qdrant collection.

        Args:
            pdf_name (str): The name of the PDF file whose embeddings should be deleted.
        """
        filter_ = models.Filter(
            must=[
                models.FieldCondition(
                    key="pdf_name",
                    match=models.MatchValue(value=pdf_name)
                )
            ]
        )

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=filter_
            )
        )

        print(f"Deleted all embeddings for PDF '{pdf_name}' from collection '{self.collection_name}'.")

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search the Qdrant collection for documents that match the given query
        using a hybrid search approach with reranking.

        Args:
            query (str): The search query.
            limit (int): The maximum number of results to return. Defaults to 10.

        Returns:
            List[Dict[str, Any]]: The search results with payload information.
        """
        clean_query = self._clean_text(query)
        
        # Step 1: Generate embeddings for the query
        dense_vectors = next(self.dense_model.query_embed(clean_query))
        sparse_vectors = next(self.sparse_model.query_embed(clean_query))
        late_vectors = next(self.late_interaction_model.query_embed(clean_query))

        # Step 2: Prepare prefetch sub-queries (for hybrid search)
        prefetch = [
            models.Prefetch(
                query=dense_vectors,
                using="dense-vector",
                limit=20,
            ),
            models.Prefetch(
                query=models.SparseVector(**sparse_vectors.as_object()),
                using="bm25",
                limit=20,
            ),
        ]

        # Step 3: Perform the search with reranking
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=late_vectors,
            using="output-token-embeddings",
            with_payload=True,
            limit=limit,
        )
        
        # Step 4: Format and return the results
        formatted_results = []
        for res in results:
            formatted_results.append({
                "id": res.id,
                "payload": res.payload,
                "score": res.score
            })

        return formatted_results
    
    def get_document_by_page(self, pdf_name: str, page_number: int):
        """
        Retrieve a specific page from a PDF document directly by metadata.

        Args:
            pdf_name (str): Name of the PDF document
            page_number (int): Page number to retrieve

        Returns:
            Dict or None: Document payload if found, None otherwise
        """
        filter_ = models.Filter(
            must=[
                models.FieldCondition(
                    key="pdf_name",
                    match=models.MatchValue(value=pdf_name)
                ),
                models.FieldCondition(
                    key="page_number",
                    match=models.MatchValue(value=page_number)
                )
            ]
        )
        
        results = self.client.scroll(
            collection_name=self.collection_name,
            filter=filter_,
            limit=1,
            with_payload=True
        )
        
        points = results[0]
        if points:
            return points[0].payload
        return None

    def __str__(self):
        """
        Return a string representation of the EnhancedBestRAG object.
        """
        info = (
            "**************************************************\n"
            "* EnhancedBestRAG Object Information             *\n"
            "**************************************************\n"
            f"* Local Storage Path: {self.persistence_path}\n"
            f"* Collection Name: {self.collection_name}\n"
            "**************************************************"
        )
        return info
