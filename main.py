import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# Load datasets
@st.cache_data
def load_data():
    hadith_df = pd.read_csv('hadiths.csv')
    quran_df = pd.read_csv('quran.csv')
    return hadith_df, quran_df

hadith_df, quran_df = load_data()

# Initialize the Sentence Transformer model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Generate embeddings for Hadith and Quran data
hadith_embeddings = model.encode(hadith_df['text_column'].tolist())
quran_embeddings = model.encode(quran_df['text_column'].tolist())

# Create FAISS index
dimension = hadith_embeddings.shape[1]  # The size of the embeddings
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the index
index.add(hadith_embeddings)
index.add(quran_embeddings)

# Streamlit UI
st.title("Islamic Text Search Engine")

query = st.text_input("Enter your query:")

if query:
    def search(query, k=5):
        # Generate embedding for the query
        query_embedding = model.encode([query])
        
        # Search in FAISS
        distances, indices = index.search(query_embedding, k)
        
        return distances, indices

    distances, indices = search(query, k=5)

    st.write("Distances:", distances)
    st.write("Indices:", indices)

    # Fetch the corresponding texts
    def get_texts_from_indices(df, indices):
        texts = []
        for idx in indices.flatten():
            if idx < len(df):  # Ensure index is within bounds
                texts.append(df.iloc[idx]['text_column'])
        return texts

    hadith_texts = get_texts_from_indices(hadith_df, indices)
    quran_texts = get_texts_from_indices(quran_df, indices)

    st.write("Hadith Texts:")
    st.write(hadith_texts)
    st.write("Quran Texts:")
    st.write(quran_texts)
