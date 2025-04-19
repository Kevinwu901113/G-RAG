import asyncio
import json
import os
from typing import Dict, List, Optional, Union, Any, Callable

from ..core.base import QueryParam
from ..utils.common import logger
from .lightrag import LightRAG


async def query_with_mode(
    rag: LightRAG,
    query: str,
    mode: str = "global",
    response_type: str = "Multiple Paragraphs",
    top_k: int = 60,
    max_token_for_text_unit: int = 4000,
    max_token_for_global_context: int = 4000,
    max_token_for_local_context: int = 4000,
    only_need_context: bool = False,
    **kwargs,
) -> Union[str, Dict[str, Any]]:
    """
    Query the RAG system with a specific mode.
    
    Args:
        rag: The LightRAG instance
        query: The query string
        mode: The query mode, one of "local", "global", "hybrid", "naive"
        response_type: The response type, e.g., "Multiple Paragraphs", "Bullet Points", etc.
        top_k: Number of top-k items to retrieve
        max_token_for_text_unit: Maximum tokens for text chunks
        max_token_for_global_context: Maximum tokens for global context
        max_token_for_local_context: Maximum tokens for local context
        only_need_context: Whether to only return the context without generating a response
        **kwargs: Additional arguments to pass to the LLM
        
    Returns:
        The query response or context
    """
    param = QueryParam(
        mode=mode,
        response_type=response_type,
        top_k=top_k,
        max_token_for_text_unit=max_token_for_text_unit,
        max_token_for_global_context=max_token_for_global_context,
        max_token_for_local_context=max_token_for_local_context,
        only_need_context=only_need_context,
    )
    
    return await rag.aquery(query, param=param, **kwargs)


async def batch_query(
    rag: LightRAG,
    queries: List[str],
    mode: str = "global",
    response_type: str = "Multiple Paragraphs",
    top_k: int = 60,
    max_token_for_text_unit: int = 4000,
    max_token_for_global_context: int = 4000,
    max_token_for_local_context: int = 4000,
    only_need_context: bool = False,
    max_concurrent: int = 5,
    **kwargs,
) -> List[Union[str, Dict[str, Any]]]:
    """
    Execute multiple queries in parallel.
    
    Args:
        rag: The LightRAG instance
        queries: List of query strings
        mode: The query mode, one of "local", "global", "hybrid", "naive"
        response_type: The response type, e.g., "Multiple Paragraphs", "Bullet Points", etc.
        top_k: Number of top-k items to retrieve
        max_token_for_text_unit: Maximum tokens for text chunks
        max_token_for_global_context: Maximum tokens for global context
        max_token_for_local_context: Maximum tokens for local context
        only_need_context: Whether to only return the context without generating a response
        max_concurrent: Maximum number of concurrent queries
        **kwargs: Additional arguments to pass to the LLM
        
    Returns:
        List of query responses or contexts
    """
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def query_with_semaphore(query: str) -> Union[str, Dict[str, Any]]:
        async with semaphore:
            return await query_with_mode(
                rag=rag,
                query=query,
                mode=mode,
                response_type=response_type,
                top_k=top_k,
                max_token_for_text_unit=max_token_for_text_unit,
                max_token_for_global_context=max_token_for_global_context,
                max_token_for_local_context=max_token_for_local_context,
                only_need_context=only_need_context,
                **kwargs,
            )
    
    # Execute queries in parallel with concurrency limit
    tasks = [query_with_semaphore(query) for query in queries]
    results = await asyncio.gather(*tasks)
    
    return results


async def compare_query_modes(
    rag: LightRAG,
    query: str,
    modes: List[str] = ["local", "global", "hybrid", "naive"],
    response_type: str = "Multiple Paragraphs",
    top_k: int = 60,
    max_token_for_text_unit: int = 4000,
    max_token_for_global_context: int = 4000,
    max_token_for_local_context: int = 4000,
    **kwargs,
) -> Dict[str, str]:
    """
    Compare different query modes for the same query.
    
    Args:
        rag: The LightRAG instance
        query: The query string
        modes: List of modes to compare
        response_type: The response type
        top_k: Number of top-k items to retrieve
        max_token_for_text_unit: Maximum tokens for text chunks
        max_token_for_global_context: Maximum tokens for global context
        max_token_for_local_context: Maximum tokens for local context
        **kwargs: Additional arguments to pass to the LLM
        
    Returns:
        Dictionary mapping mode names to responses
    """
    results = {}
    
    for mode in modes:
        logger.info(f"Querying with mode: {mode}")
        response = await query_with_mode(
            rag=rag,
            query=query,
            mode=mode,
            response_type=response_type,
            top_k=top_k,
            max_token_for_text_unit=max_token_for_text_unit,
            max_token_for_global_context=max_token_for_global_context,
            max_token_for_local_context=max_token_for_local_context,
            **kwargs,
        )
        results[mode] = response
    
    return results


async def evaluate_query(
    rag: LightRAG,
    query: str,
    reference_answer: str,
    mode: str = "global",
    response_type: str = "Multiple Paragraphs",
    top_k: int = 60,
    max_token_for_text_unit: int = 4000,
    max_token_for_global_context: int = 4000,
    max_token_for_local_context: int = 4000,
    **kwargs,
) -> Dict[str, Any]:
    """
    Evaluate a query against a reference answer.
    
    Args:
        rag: The LightRAG instance
        query: The query string
        reference_answer: The reference answer to compare against
        mode: The query mode
        response_type: The response type
        top_k: Number of top-k items to retrieve
        max_token_for_text_unit: Maximum tokens for text chunks
        max_token_for_global_context: Maximum tokens for global context
        max_token_for_local_context: Maximum tokens for local context
        **kwargs: Additional arguments to pass to the LLM
        
    Returns:
        Evaluation results including the generated answer and evaluation metrics
    """
    # Get the generated answer
    generated_answer = await query_with_mode(
        rag=rag,
        query=query,
        mode=mode,
        response_type=response_type,
        top_k=top_k,
        max_token_for_text_unit=max_token_for_text_unit,
        max_token_for_global_context=max_token_for_global_context,
        max_token_for_local_context=max_token_for_local_context,
        **kwargs,
    )
    
    # Get the context for analysis
    context = await query_with_mode(
        rag=rag,
        query=query,
        mode=mode,
        response_type=response_type,
        top_k=top_k,
        max_token_for_text_unit=max_token_for_text_unit,
        max_token_for_global_context=max_token_for_global_context,
        max_token_for_local_context=max_token_for_local_context,
        only_need_context=True,
        **kwargs,
    )
    
    # Evaluate the answer using the LLM
    from ..core.prompt import PROMPTS
    
    evaluation_prompt = PROMPTS["norag_response"].format(
        question=query,
        answer=reference_answer,
        answer1=generated_answer,
    )
    
    evaluation_response = await rag.llm_model_func(
        evaluation_prompt, **rag.llm_model_kwargs
    )
    
    # Extract the score from the evaluation response
    try:
        score_line = evaluation_response.strip().split('\n')[0]
        score = int(score_line.split('分')[0])
    except (ValueError, IndexError):
        score = 0
        logger.error(f"Failed to parse evaluation score from: {evaluation_response}")
    
    # Analyze which parts of the context contributed to the answer
    analysis_prompt = PROMPTS["norag_response1"].format(
        question=query,
        answer=generated_answer,
        context_data=context,
    )
    
    analysis_response = await rag.llm_model_func(
        analysis_prompt, **rag.llm_model_kwargs
    )
    
    return {
        "query": query,
        "reference_answer": reference_answer,
        "generated_answer": generated_answer,
        "context": context,
        "evaluation_score": score,
        "evaluation_details": evaluation_response,
        "context_analysis": analysis_response,
    }
