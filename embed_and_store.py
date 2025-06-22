from langchain_community.document_loaders import DirectoryLoader

from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

embedding_dim = len(embeddings.embed_query("hello world"))
index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

loader = DirectoryLoader("output_all", glob="**/*.md", show_progress=True)
docs = loader.load()
print(f"Loaded {len(docs)} documents")


ids = vector_store.add_documents(documents=docs)
print(f"Added {len(ids)} documents to vector store")

vector_store.save_local("faiss_index")

#When you ride, your gear is "right" if it protects you. In any crash, you have a far better chance of avoiding serious injury if you wear:
# 1. An approved helmet
# 2. Face or eye protection
# 3. Protective clothing
results = vector_store.similarity_search(
    "When you ride, your gear is 'right' if it protects you.",
)

print(results[0])

