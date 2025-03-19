from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm

THRESHOLD = 0.7

# https://huggingface.co/sentence-transformers/all-mpnet-base-v2
model = SentenceTransformer('all-mpnet-base-v2')

data = ""
with open(input("Filepath: "), "r") as file:
    for line in file:
        data += line

print()

sent_tokens = sent_tokenize(data, language="english")
embeddings = model.encode(sent_tokens)

scores = []

i = 0
while i < len(embeddings):
    vec_a = embeddings[i]

    j = i + 1
    while j < len(embeddings):
        vec_b = embeddings[j]
        similarity = dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b))
        if similarity >= THRESHOLD:
            scores.append((similarity, i, j))
        j += 1
    i += 1


scores.sort(key=lambda a: a[0], reverse=True)
for similarity, i, j in scores:
    print(f"Similarity score {round(similarity, 5)} between:\n\t\"{sent_tokens[i]}\"\n\t\"{sent_tokens[j]}\".")
