from dotenv import load_dotenv
import os
from flask import Flask, jsonify, request

app = Flask(__name__)

load_dotenv()

# Environment Variables
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=pinecone_api_key)
index_name = "test2"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Ensure this matches your OpenAI model's dimensionality
        metric="euclidean",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# Initialize OpenAI Embeddings
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Function to generate query vector
def generate_query_vector(text):
    return embeddings.embed_documents([text])[0]

# Function to query Pinecone
def query_pinecone(query_text):
    query_vector = generate_query_vector(query_text)
    results = index.query(
        namespace="ns1",
        vector=query_vector,
        top_k=2,
        include_values=True,
        include_metadata=True,
        # filter={"genre": {"$eq": "action"}}
    )

    extracted_text = [
        item['metadata'].get('text', None) for item in results['matches']
    ]

    return extracted_text

# Test the query

query_text = "example query text"
results = query_pinecone(query_text)
print(results)


@app.route('/')
def home():
    return jsonify(message="Hello, API!")

@app.route('/query_text', methods=['POST'])
def query_text(text: str):
    return query_pinecone(text)

if __name__ == '__main__':
    app.run(debug=True)  # Set debug=False for production