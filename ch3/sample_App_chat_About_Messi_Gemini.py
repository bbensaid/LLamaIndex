import os
from dotenv import load_dotenv

load_dotenv()  # This loads the .env file

api_key = os.getenv("GOOGLE_API_KEY") # Now you can use it
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Please create a .env file with your key.")

#######

from llama_index.llms.google_genai import GoogleGenAI

# It automatically grabs the key from os.environ["GOOGLE_API_KEY"]
llm = GoogleGenAI(model="models/gemini-flash-latest", api_key=api_key)

#######

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# For the Language Model (LLM), use your best general model
Settings.llm = llm

# For the Embedding Model, use the specialized embedding model
Settings.embed_model = GoogleGenAIEmbedding(model="models/text-embedding-004", api_key=api_key)

# --- End of Configuration ---


loader = WikipediaReader()
documents = loader.load_data(pages=["Paul Volker"])
parser = SimpleNodeParser.from_defaults()
nodes = parser.get_nodes_from_documents(documents)
index = VectorStoreIndex(nodes)
query_engine = index.as_query_engine()
print(f"Using LLM model: {Settings.llm.metadata.model_name}")
print("Ask me anything about Lionel Messi!")

while True:
    question = input("Your question: ")
    if question.lower() == "exit":
        break
    response = query_engine.query(question)
    print(response)
