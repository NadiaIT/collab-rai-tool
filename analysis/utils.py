import dspy
import litellm
from typing import List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.cluster import KMeans
import os
import numpy as np
import pandas as pd
import tqdm
import copy
import time
import json
from dotenv import load_dotenv

load_dotenv(override=True)

LM_DICT = {
    "gpt-4o": dspy.LM('openai/gpt-4o-2024-08-06', temperature=1.0, max_tokens=4096),
    "gpt-4o-2024-05-13": dspy.LM('openai/gpt-4o-2024-05-13', temperature=1.0, max_tokens=4096),
    "gpt-4o-2024-08-06": dspy.LM('openai/gpt-4o-2024-08-06', temperature=1.0, max_tokens=4096),
    "gpt-4o-2024-11-20": dspy.LM('openai/gpt-4o-2024-11-20', temperature=1.0, max_tokens=4096),
    "gpt-4o-mini": dspy.LM('openai/gpt-4o-mini-2024-07-18', temperature=1.0, max_tokens=4096),
    "4o-eval": dspy.LM('openai/gpt-4o-2024-08-06', temperature=0, max_tokens=16384),
    "4o-mini-eval": dspy.LM('openai/gpt-4o-mini-2024-07-18', temperature=0, max_tokens=16384),
    "4.1-mini-eval": dspy.LM('openai/gpt-4.1-mini', temperature=0, max_tokens=16384),
    # "o3-mini": dspy.LM('openai/o3-mini', temperature=1.0, max_tokens=10000),
    "llama3-70b": dspy.LM('bedrock/meta.llama3-70b-instruct-v1:0', temperature=0.6, max_tokens=2048),
    "llama3.1-70b": dspy.LM('bedrock/us.meta.llama3-1-70b-instruct-v1:0', temperature=0.6, max_tokens=4096),
    "llama3-2-90b-instruct": dspy.LM('bedrock/us.meta.llama3-2-90b-instruct-v1:0', temperature=0.6, max_tokens=4096),
    "llama3.3-70b": dspy.LM('bedrock/us.meta.llama3-3-70b-instruct-v1:0', temperature=0.6, max_tokens=4096),
}


def use_lm(lm, n=1):
    def decorator(program):
        def wrapper(*args, **kwargs):
            max_retries = 3
            initial_delay = 1
            delay = initial_delay
            
            for attempt in range(max_retries):
                try:
                    with dspy.context(lm=lm):
                        return program(*args, **kwargs)
                except litellm.APIError as e:
                    if attempt < max_retries - 1:
                        print(f"API Error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                    else:
                        raise
                except Exception as e:
                    print(f"Error: {e}")
                    return dspy.Example(output="")
        return wrapper
    return decorator

def batch_inference(program, args_list, max_workers=32) -> List[Any]:
    futures = {}
    results = [None] * len(args_list)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, args in enumerate(args_list):
            future = executor.submit(
                program,
                **args
            )
            futures[future] = i

        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            index = futures[future]
            results[index] = result
    return results

def run_model(program, examples, max_workers=32):
    examples = copy.deepcopy(examples)
    results = batch_inference(
        program,
        [example.inputs().toDict() for example in examples],
        max_workers=max_workers,
    )
    for example, result in zip(examples, results):
        example.output = result.output
        example.outputs = result.outputs
    return examples

def get_embeddings(texts, embedding_model='openai/text-embedding-ada-002'):
    return batch_inference(
        lambda text: litellm.embedding(model=embedding_model, input=[text]).data[0]['embedding'],
        [{"text": text} for text in texts]
    )