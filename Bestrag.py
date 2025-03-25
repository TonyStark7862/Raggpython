"""EnhancedBestRAG"""
# Authors: Based on work by Abdul Samad Siddiqui <abdulsamadsid1@gmail.com>

import re
import uuid
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance
from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding
import PyPDF2


class EnhancedBestRAG:
    """
    EnhancedBestRAG is an improved version of BestRAG that uses local Qdrant persistence
    and implements the hybrid search approach with reranking as shown in the Qdrant documentation.

    It supports dense embeddings (all-MiniLM-L6-v2), sparse embeddings (bm25),
    and ColBERT late interaction model for reranking.

    Args:
        collection_name (str): The name of the Qdrant collection to use.
        persistence_path (str, optional): Path for local Qdrant storage. Defaults to "./qdrant_data".
    """

    def __init__(self,
                 collection_name: str,
                 persistence_path: str = "./qdrant_data",
                 dense_model_path: str = "sentence-transformers/all-MiniLM-L6-v2",
                 bm25_model_path: str = "Qdrant/bm25",
                 colbert_model_path: str = "colbert-ir/colbertv2.0"
                 ):
        self.collection_name = collection_name
        self.persistence_path = persistence_path
        
        # Store model paths
        self.dense_model_path = dense_model_path
        self.bm25_model_path = bm25_model_path
        self.colbert_model_path = colbert_model_path
        
        # Define vector field names (consistent regardless of model path)
        self.dense_field_name = "dense-vector"
        self.sparse_field_name = "sparse-vector"
        self.late_field_name = "late-vector"
        
        # Initialize local Qdrant client
        self.client = QdrantClient(path=persistence_path)

        # Initialize embedding models with provided paths
        self.dense_embedding_model = TextEmbedding(dense_model_path)
        self.bm25_embedding_model = SparseTextEmbedding(bm25_model_path)
        self.late_interaction_embedding_model = LateInteractionTextEmbedding(colbert_model_path)

        self._create_or_use_collection()

    def _create_or_use_collection(self):
        """
        Create a new Qdrant collection if it doesn't exist, or use an existing one.
        Uses consistent vector field names regardless of model path.
        """
        collections = self.client.get_collections()
        collection_names = [
            collection.name for collection in collections.collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    self.dense_field_name: models.VectorParams(
                        size=384,
                        distance=Distance.COSINE
                    ),
                    self.late_field_name: models.VectorParams(
                        size=384,
                        distance=Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        )
                    ),
                },
                sparse_vectors_config={self.sparse_field_name: models.SparseVectorParams()}
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
        Implements the approach shown in the documentation images.

        Args:
            pdf_path (str): The path to the PDF file.
            pdf_name (str): The name of the PDF file.
            metadata (Optional[dict]): Additional metadata to store with each embedding.
        """
        texts = self._extract_pdf_text_per_page(pdf_path)
        points = []

        for page_num, text in enumerate(texts):
            # No cleaning as requested
            doc_text = text
            
            # Generate all embeddings
            dense_embedding = list(self.dense_embedding_model.embed([doc_text]))[0]
            bm25_embedding = next(self.bm25_embedding_model.embed(doc_text))
            late_interaction_embedding = list(self.late_interaction_embedding_model.embed([doc_text]))[0]

            # Prepare document payload
            payload = {
                "document": doc_text,
                "page_number": page_num + 1,
                "pdf_name": pdf_name
            }
            
            # Add any additional metadata
            if metadata:
                payload.update(metadata)

            # Create point with consistent vector field names
            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    self.dense_field_name: dense_embedding,
                    self.sparse_field_name: bm25_embedding.as_object(),
                    self.late_field_name: late_interaction_embedding,
                },
                payload=payload
            )
            
            points.append(point)

        # Batch upsert all points
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"Stored embeddings for {len(texts)} pages of '{pdf_name}' in collection '{self.collection_name}'.")

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
        Search the Qdrant collection using the hybrid search approach shown in the documentation.
        
        This implements the exact approach from the screenshots with:
        1. Query embedding generation
        2. Prefetch setup with both dense and sparse vectors
        3. Final search with reranking using the late interaction model
        
        Args:
            query (str): The search query.
            limit (int): The maximum number of results to return. Defaults to 10.

        Returns:
            List[Dict[str, Any]]: The search results with payload information.
        """
        # Step 1: Generate embeddings for the query (exactly as shown in Image 6)
        dense_vectors = next(self.dense_embedding_model.query_embed(query))
        sparse_vectors = next(self.bm25_embedding_model.query_embed(query))
        late_vectors = next(self.late_interaction_embedding_model.query_embed(query))

        # Step 2: Prepare prefetch for hybrid search with consistent field names
        prefetch = [
            models.Prefetch(
                query=dense_vectors,
                using=self.dense_field_name,
                limit=20,
            ),
            models.Prefetch(
                query=models.SparseVector(**sparse_vectors.as_object()),
                using=self.sparse_field_name,
                limit=20,
            ),
        ]

        # Step 3: Execute query with reranking using consistent field names
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=late_vectors,
            using=self.late_field_name,
            with_payload=True,
            limit=limit,
        )
        
        # Format results
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
