from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


import logging
logging.basicConfig(level=logging.WARNING)

# Configure Ollama (Llama 3.1) and Nomic Embeddings
Settings.llm = Ollama(model="llama3.1", request_timeout=800.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

loader = WikipediaReader()
documents = loader.load_data(pages=["Messi Lionel"])
parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = parser.get_nodes_from_documents(documents)
index = VectorStoreIndex(nodes)
query_engine = index.as_query_engine(similarity_top_k=3, streaming=True)
print(f"Using LLM model: {Settings.llm.metadata.model_name}")
print("Ask me anything about Lionel Messi!")

while True:
    question = input("Your question: ")
    if question.lower() == "exit":
        break
    response = query_engine.query(question)
    response.print_response_stream()