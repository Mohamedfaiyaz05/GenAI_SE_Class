from langchain_huggingface import HuggingFaceEmbeddings

# A small, free, popular embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Embedding model is ready!")

text = "I love programming in Python."

vector = embeddings.embed_query(text)

print("How many numbers in this vector?", len(vector))
print("First 5 numbers:", vector[:5])
print(vector)  # 

import numpy as np

def similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

v1 = embeddings.embed_query("I like dogs")
v2 = embeddings.embed_query("I love puppies")
v3 = embeddings.embed_query("The car is fast")

print("dogs vs puppies (similar):", round(similarity(v1, v2), 2))
print("dogs vs car   (different):", round(similarity(v1, v3), 2))