import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient

load_dotenv()

# load the embedding model
model_name = "BAAI/bge-large-en"
model_kwargs = { 'device': 'cpu'}
encode_kwargs = { 'normalize_embeddings': False }

embeddings = HuggingFaceBgeEmbeddings(
    model_name = model_name, 
    model_kwargs = model_kwargs, 
    encode_kwargs = encode_kwargs
)

qdrant_url = os.environ.get("QDRANT_DB_ENDPOINT_URL")
collection_name = os.environ.get("COLLECTION_NAME")

client = QdrantClient(
    url = qdrant_url,
    prefer_grpc= False,
)

db = Qdrant(
    client = client,
    embeddings = embeddings,
    collection_name =collection_name
)

query = "What are some of the limitations of GPT-4?"

docs = db.similarity_search_with_score(query=query, k=5)
for data in docs:
    doc, score = data
    print("score:" , score, "\ncontent:" , doc.page_content, "\nmetadata:" , doc.metadata, "\n")