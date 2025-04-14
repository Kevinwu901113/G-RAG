import os
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc
from unstructured.partition.docx import partition_docx
import re
import json

WORKING_DIR = "./result/test3"
OLLAMA_SCHED_SPREAD = 1
OLLAMA_FLASH_ATTENTION = 1
CUDA_VISIBLE_DEVICES = 1

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="qwen2.5:7b-instruct-fp16",
    # llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=lambda texts: ollama_embedding(
            texts, embed_model="bge-m3", host="http://localhost:11434"
        ),
    ),
)

data_folder = r'./data'
text_content = []
for root, dirs, files in os.walk(data_folder):
    for file in files:
        if file.endswith('.docx'):
            file_path = os.path.join(root, file)
            elements = partition_docx(file_path)
            text = "\n".join([str(e) for e in elements])
            text_content.append(text)

rag.insert(text_content)

# print(
#     rag.query("这篇文章包含了哪些主题", param=QueryParam(mode="local"))
# )
#
# # global模式
# print(
#     rag.query("这篇文章包含了哪些主题", param=QueryParam(mode="global"))
# )

# hybrid模式
# tem = {"question": "", "ans": "", "source": []}
# question = "公明区对外经济引进小组是什么时候成立的？"
# ans, sour = rag.query(question, param=QueryParam(mode="hybrid"))
# lines = sour.split('\n')
# sources = []
# pattern = re.compile(r'^\d,')
# f = 0
# for line in lines:
#     match = pattern.match(line)
#     if match:
#         sources.append(line)
#         f = 1
#     elif f == 1:
#         sources[-1] += line
# tem["question"] = question
# tem["ans"] = ans

# for source in sources:
#     print(source)
#     ans1 = rag.query1(ans, source, question)
#     tem["source"].append(ans1)

# output_file_path = 'other.json'
# with open(output_file_path, 'a', encoding='utf-8') as f:
#     f.write(json.dumps(tem, ensure_ascii=False) + "\n")
