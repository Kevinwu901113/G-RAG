import asyncio
import os
from typing import List, Dict, Any, Optional, Callable, Union

import numpy as np

from ..utils.common import logger, wrap_embedding_func_with_attrs


class EmbeddingManager:
    """
    Manages embedding functions and provides utilities for embedding text.
    """
    
    def __init__(self, default_embedding_func: Optional[Callable] = None):
        """
        Initialize the embedding manager.
        
        Args:
            default_embedding_func: The default embedding function to use
        """
        self.default_embedding_func = default_embedding_func
        self.embedding_funcs = {}
        
        if default_embedding_func:
            self.register_embedding_func("default", default_embedding_func)
    
    def register_embedding_func(self, name: str, func: Callable) -> None:
        """
        Register an embedding function with a name.
        
        Args:
            name: The name to register the function under
            func: The embedding function
        """
        self.embedding_funcs[name] = func
        logger.info(f"Registered embedding function: {name}")
    
    async def embed_text(
        self, 
        texts: List[str], 
        func_name: Optional[str] = None,
        batch_size: int = 32,
        **kwargs
    ) -> np.ndarray:
        """
        Embed a list of texts using the specified embedding function.
        
        Args:
            texts: List of texts to embed
            func_name: Name of the embedding function to use (uses default if None)
            batch_size: Size of batches to process
            **kwargs: Additional arguments to pass to the embedding function
            
        Returns:
            Array of embeddings
        """
        if not texts:
            return np.array([])
        
        # Select embedding function
        if func_name and func_name in self.embedding_funcs:
            embed_func = self.embedding_funcs[func_name]
        elif self.default_embedding_func:
            embed_func = self.default_embedding_func
        else:
            raise ValueError("No embedding function specified and no default available")
        
        # Process in batches
        batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
        embeddings_list = await asyncio.gather(*[embed_func(batch, **kwargs) for batch in batches])
        
        # Combine results
        return np.concatenate(embeddings_list)
    
    async def embed_and_search(
        self,
        query: str,
        corpus: List[str],
        func_name: Optional[str] = None,
        top_k: int = 5,
        batch_size: int = 32,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Embed a query and search for similar texts in a corpus.
        
        Args:
            query: The query text
            corpus: List of texts to search in
            func_name: Name of the embedding function to use
            top_k: Number of top results to return
            batch_size: Size of batches to process
            **kwargs: Additional arguments to pass to the embedding function
            
        Returns:
            List of dictionaries with text and similarity score
        """
        if not corpus:
            return []
        
        # Embed query and corpus
        query_embedding = await self.embed_text([query], func_name, batch_size, **kwargs)
        corpus_embeddings = await self.embed_text(corpus, func_name, batch_size, **kwargs)
        
        # Calculate cosine similarities
        query_embedding = query_embedding[0]
        similarities = self._cosine_similarity(query_embedding, corpus_embeddings)
        
        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = [
            {
                "text": corpus[idx],
                "score": float(similarities[idx]),
                "index": int(idx)
            }
            for idx in top_indices
        ]
        
        return results
    
    @staticmethod
    def _cosine_similarity(query_embedding: np.ndarray, corpus_embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between a query embedding and corpus embeddings.
        
        Args:
            query_embedding: The query embedding vector
            corpus_embeddings: Matrix of corpus embedding vectors
            
        Returns:
            Array of similarity scores
        """
        # Normalize vectors
        query_norm = np.linalg.norm(query_embedding)
        corpus_norms = np.linalg.norm(corpus_embeddings, axis=1)
        
        # Avoid division by zero
        query_norm = np.maximum(query_norm, 1e-10)
        corpus_norms = np.maximum(corpus_norms, 1e-10)
        
        # Calculate cosine similarity
        similarities = np.dot(corpus_embeddings, query_embedding) / (corpus_norms * query_norm)
        
        return similarities


def create_openai_embedding_func(
    model: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    embedding_dim: int = 1536,
    max_token_size: int = 8192,
) -> Callable:
    """
    Create an OpenAI embedding function.
    
    Args:
        model: The OpenAI embedding model to use
        api_key: OpenAI API key (uses environment variable if None)
        base_url: Base URL for API (uses default if None)
        embedding_dim: Dimension of embeddings
        max_token_size: Maximum token size
        
    Returns:
        Embedding function
    """
    from ..core.llm import openai_embedding
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    @wrap_embedding_func_with_attrs(embedding_dim=embedding_dim, max_token_size=max_token_size)
    async def embed_func(texts: List[str]) -> np.ndarray:
        return await openai_embedding(texts, model=model, base_url=base_url)
    
    return embed_func


def create_azure_openai_embedding_func(
    model: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    embedding_dim: int = 1536,
    max_token_size: int = 8192,
) -> Callable:
    """
    Create an Azure OpenAI embedding function.
    
    Args:
        model: The Azure OpenAI embedding model to use
        api_key: Azure OpenAI API key (uses environment variable if None)
        base_url: Base URL for API (uses environment variable if None)
        embedding_dim: Dimension of embeddings
        max_token_size: Maximum token size
        
    Returns:
        Embedding function
    """
    from ..core.llm import azure_openai_embedding
    
    if api_key:
        os.environ["AZURE_OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["AZURE_OPENAI_ENDPOINT"] = base_url
    
    @wrap_embedding_func_with_attrs(embedding_dim=embedding_dim, max_token_size=max_token_size)
    async def embed_func(texts: List[str]) -> np.ndarray:
        return await azure_openai_embedding(texts, model=model)
    
    return embed_func


def create_bedrock_embedding_func(
    model: str = "amazon.titan-embed-text-v2:0",
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    embedding_dim: int = 1536,
    max_token_size: int = 8192,
) -> Callable:
    """
    Create an Amazon Bedrock embedding function.
    
    Args:
        model: The Bedrock embedding model to use
        aws_access_key_id: AWS access key ID
        aws_secret_access_key: AWS secret access key
        aws_session_token: AWS session token
        embedding_dim: Dimension of embeddings
        max_token_size: Maximum token size
        
    Returns:
        Embedding function
    """
    from ..core.llm import bedrock_embedding
    
    @wrap_embedding_func_with_attrs(embedding_dim=embedding_dim, max_token_size=max_token_size)
    async def embed_func(texts: List[str]) -> np.ndarray:
        return await bedrock_embedding(
            texts,
            model=model,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )
    
    return embed_func


def create_ollama_embedding_func(
    model: str = "llama3",
    host: Optional[str] = None,
    embedding_dim: int = 4096,
    max_token_size: int = 8192,
) -> Callable:
    """
    Create an Ollama embedding function.
    
    Args:
        model: The Ollama model to use
        host: Ollama host URL
        embedding_dim: Dimension of embeddings
        max_token_size: Maximum token size
        
    Returns:
        Embedding function
    """
    from ..core.llm import ollama_embedding
    
    @wrap_embedding_func_with_attrs(embedding_dim=embedding_dim, max_token_size=max_token_size)
    async def embed_func(texts: List[str]) -> np.ndarray:
        kwargs = {}
        if host:
            kwargs["host"] = host
        
        return await ollama_embedding(texts, embed_model=model, **kwargs)
    
    return embed_func


def create_huggingface_embedding_func(
    model_name: str,
    tokenizer_name: Optional[str] = None,
    embedding_dim: Optional[int] = None,
    max_token_size: int = 512,
) -> Callable:
    """
    Create a Hugging Face embedding function.
    
    Args:
        model_name: The Hugging Face model to use
        tokenizer_name: The tokenizer to use (uses model_name if None)
        embedding_dim: Dimension of embeddings (determined from model if None)
        max_token_size: Maximum token size
        
    Returns:
        Embedding function
    """
    from transformers import AutoTokenizer, AutoModel
    import torch
    from ..core.llm import hf_embedding
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Determine embedding dimension if not provided
    if embedding_dim is None:
        with torch.no_grad():
            sample_input = tokenizer("Sample text", return_tensors="pt")
            sample_output = model(**sample_input)
            embedding_dim = sample_output.last_hidden_state.shape[-1]
    
    @wrap_embedding_func_with_attrs(embedding_dim=embedding_dim, max_token_size=max_token_size)
    async def embed_func(texts: List[str]) -> np.ndarray:
        return await hf_embedding(texts, tokenizer, model)
    
    return embed_func
