import streamlit as st
import pandas as pd
import pinecone
from sentence_transformers import SentenceTransformer
from ibm_watsonx_ai import WatsonXAI  # Update this based on latest API

# Initialize Pinecone
pinecone.init(api_key='bf3175a2-10b9-4758-b652-3456b4a73d3c', environment='us-west1-gcp')
index_name = 'quran-hadith-index'
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384)
index = pinecone.Index(index_name)

# Initialize the multilingual embedding model
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Initialize IBM Watsonx AI
watsonx_ai = WatsonXAI(
    api_key="Pc6M6Vuqe4gQsq0ZGfa2FaUVd6G816Emtx_-Era1UjHQ",  # Replace with your API key
    url="https://us-south.ml.cloud.ibm.com"  # Replace with your service URL
)
project_id = "835090d0-df03-46d0-ae28-e3ebeed066b9"
model_id = "ibm/granite-13b-chat-v2"
parameters = {
    "decoding_method": "sample",
    "max_new_tokens": 600,
    "temperature": 0.2,
    "top_k": 1,
    "top_p": 1
}
model = watsonx_ai.model(
    model_id=model_id,
    params=parameters,
    project_id=project_id
)

def process_chunk(chunk):
    # Your processing logic here
    return chunk

# Load datasets (change file paths as needed)
def load_datasets():
    quran_df = pd.read_csv('quran.csv', chunksize=1000)
    hadith_df = pd.read_csv('hadiths.csv', chunksize=1000)
    processed_quran = pd.concat([process_chunk(chunk) for chunk in quran_df])
    processed_hadith = pd.concat([process_chunk(chunk) for chunk in hadith_df])
    return processed_quran, processed_hadith

# Index data
def index_data(processed_quran, processed_hadith):
    for chunk in processed_quran:
        embeddings = embedding_model.encode(chunk['text'].tolist())
        ids = [f'quran-{i}' for i in range(len(embeddings))]
        index.upsert(vectors=zip(ids, embeddings))

    for chunk in processed_hadith:
        embeddings = embedding_model.encode(chunk['text'].tolist())
        ids = [f'hadith-{i}' for i in range(len(embeddings))]
        index.upsert(vectors=zip(ids, embeddings))

# Query Pinecone
def query_pinecone(query):
    query_embedding = embedding_model.encode([query])
    result = index.query(query_embedding, top_k=5)
    return result['matches']

# Generate response using IBM Watsonx
def generate_response(prompt):
    response = model.generate(prompt)
    return response['generated_text']

# Streamlit UI
st.title('Quran and Hadith Query System')

query = st.text_input("Enter your query:")
if st.button('Submit'):
    if query:
        processed_quran, processed_hadith = load_datasets()
        index_data(processed_quran, processed_hadith)
        
        if 'hadith' in query.lower():
            matches = query_pinecone(query)
            response = f"Hadith Matches: {matches}"
        else:
            matches = query_pinecone(query)
            response = f"Quran Matches: {matches}"
        
        additional_response = generate_response(query)
        final_response = f"{response}\n\n{additional_response}"
        
        st.write(final_response)
    else:
        st.write("Please enter a query.")
