from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = "true"

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=pinecone_api_key)

index_name = "test2"

# Check if the index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Replace with your model's embedding dimension
        metric="euclidean",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

# EMBED
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

text = "This is an example sentence that we want to convert into a vector."
year = 2020

vector = embeddings.embed_documents([text])[0]

print(vector) 

index.upsert(
    vectors=[
        {
            "id": "vec1", 
            "values": vector, 
            "metadata": {"year": year}
        }
    ],
    namespace= "ns1"
)

def query(vect):
    print(index.query(
        namespace="ns1",
        vector=vect,
        top_k=2,
        include_values=True,
        include_metadata=True,
        filter={"genre": {"$eq": "action"}}
    ))

query(vector)
