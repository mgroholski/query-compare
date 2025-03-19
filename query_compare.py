from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm
import os
import heapq

TOP_K = 10

# https://huggingface.co/sentence-transformers/all-mpnet-base-v2
model = SentenceTransformer('all-mpnet-base-v2')

tokens = []

# TODO: Change all command to data/*
inp_filepath = input("Filepath (all): ")
filepaths =[]
if inp_filepath.lower() == "all":
    filepaths = []
    for file in os.listdir("data/"):
        filepaths.append(os.path.join("data/", file))
else:
    filepaths.append(inp_filepath)

for filepath in filepaths:
    data = ""
    with open(filepath, "r") as file:
        for line in file:
            data += line

    tokens += sent_tokenize(data, language="english")

embeddings = model.encode(tokens)

query = input("Query: ")
query_tokens = sent_tokenize(query, language="english")
query_embeddings = model.encode(query_tokens)

scores = []
for idx, embedding in enumerate(embeddings):
    score = 0

    for q_embedding in query_embeddings:
        score += (dot(embedding, q_embedding)) / (norm(embedding) * norm(q_embedding))

    score /= len(query_tokens)
    heapq.heappush(scores, (-1 * score, idx))

i = 0
while i < TOP_K:
    rank_idx, (score, idx) =  i + 1, heapq.heappop(scores)
    print(f"Rank {rank_idx} Token:\n\tToken: {tokens[idx]}\n\tSimilarity Score: {-1 * score}")
    i += 1
