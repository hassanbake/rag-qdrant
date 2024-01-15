import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

loader = PyPDFLoader("data.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

texts = text_splitter.split_documents(documents)

# load the embedding model
model_name = "BAAI/bge-large-en"
model_kwargs = { 'device': 'cpu'}
encode_kwargs = { 'normalize_embeddings': False }

embeddings = HuggingFaceBgeEmbeddings(
    model_name = model_name, 
    model_kwargs = model_kwargs, 
    encode_kwargs = encode_kwargs
)

print("Embeddings Model Loaded...")

qdrant_url = os.environ.get("QDRANT_DB_ENDPOINT_URL")
collection_name = "gpt_collection"

qdrant = Qdrant.from_documents(
    texts = texts,
    embeddings = embeddings,
    url = qdrant_url,
    prefer_grpd = False,
    collection_name = collection_name
)

print("Qdrant Index Created...")

