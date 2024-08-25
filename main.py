import pandas as pd

# Load the Quran and Hadith datasets
quran_df = pd.read_csv('quran.csv', chunksize=1000)
hadith_df = pd.read_csv('hadiths.csv', chunksize=1000)

# Function to process each chunk
def process_chunk(chunk):
    # Your processing logic here
    return chunk

# Process the Quran dataset in chunks
for chunk in quran_df:
    processed_quran = process_chunk(chunk)
    
# Process the Hadith dataset in chunks
for chunk in hadith_df:
    processed_hadith = process_chunk(chunk)

import pinecone
from sentence_transformers import SentenceTransformer

# Initialize Pinecone
pinecone.init(api_key='bf3175a2-10b9-4758-b652-3456b4a73d3c', environment='us-west1-gcp')

# Create or connect to an index
index_name = 'quran-hadith-index'
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384)

index = pinecone.Index(index_name)

# Initialize the multilingual embedding model
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Example of indexing data
for chunk in processed_quran:
    embeddings = embedding_model.encode(chunk['text'].tolist())
    ids = [f'quran-{i}' for i in range(len(embeddings))]
    index.upsert(vectors=zip(ids, embeddings))

for chunk in processed_hadith:
    embeddings = embedding_model.encode(chunk['text'].tolist())
    ids = [f'hadith-{i}' for i in range(len(embeddings))]
    index.upsert(vectors=zip(ids, embeddings))

from ibm_watsonx_ai.foundation_models import Model, Credentials

# Set up IBM Watsonx credentials
credentials = Credentials(
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

model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)

# Example function to generate response
def generate_response(prompt):
    response = model.generate(prompt)
    return response['generated_text']

# Example of using the model
prompt = "Who is Adam A.S.?"
response = generate_response(prompt)
print(response)

def query_pinecone(query):
    # Encode the query using the multilingual model
    query_embedding = embedding_model.encode([query])
    
    # Query Pinecone for similar vectors
    result = index.query(query_embedding, top_k=5)
    
    return result['matches']

def get_quran_response(query):
    # Example function to get a Quranic response
    matches = query_pinecone(query)
    # Combine results and generate a response
    # You can add logic to format it as required
    return matches

def get_hadith_response(query):
    # Example function to get a Hadith response
    matches = query_pinecone(query)
    # Combine results and generate a response
    return matches

def get_combined_response(query):
    # Check if the query is related to Hadith
    if 'hadith' in query.lower():
        response = get_hadith_response(query)
    else:
        response = get_quran_response(query)
    
    # Use IBM Watsonx model for additional response generation
    additional_response = generate_response(query)
    
    # Combine the results
    final_response = f"{response}\n\n{additional_response}"
    
    return final_response

# Example usage
query = "Tell me about Adam A.S."
response = get_combined_response(query)
print(response)