import faiss

index = faiss.read_index("/home/dungx/LGI/rag/storage/faiss.index")

print("Total vectors:", index.ntotal)
print("Vector dimension:", index.d)
print("Index type:", type(index))
print(index)