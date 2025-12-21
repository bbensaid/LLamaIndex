""" from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader('files').load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("summarize each document in a few sentences")

print(response)
 """


from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# --- Recommended Configuration ---

# For the Language Model (LLM), use your best general model
Settings.llm = Ollama(model="llama3.1", request_timeout=800.0) # Increased timeout for complex queries

# For the Embedding Model, use the specialized embedding model
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# --- End of Configuration ---


# The rest of your code will now use this optimal combination
documents = SimpleDirectoryReader("files").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

print("Querying with Llama 3.1 and Nomic Embeddings...")
response = query_engine.query("summarize each document in a few sentences")
print(response)

