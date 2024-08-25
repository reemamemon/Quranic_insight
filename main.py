import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# Load datasets with error handling
@st.cache_data
def load_data():
    try:
        hadith_df = pd.read_csv('hadiths.csv', encoding='utf-8', error_bad_lines=False, warn_bad_lines=True)
        quran_df = pd.read_csv('quran.csv', encoding='utf-8', error_bad_lines=False, warn_bad_lines=True)
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        return None, None
    return hadith_df, quran_df

hadith_df, quran_df = load_data()

if hadith_df is not None and quran_df is not None:
    # Initialize the Sentence Transformer model
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    # Generate embeddings for Hadith and Quran data
    hadith_embeddings = model.encode(hadith_df['text_column'].tolist())
    quran_embeddings = model.encode(quran_df['text_column'].tolist())

    # Create FAISS index
    dimension = hadith_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(hadith_embeddings)
    index.add(quran_embeddings)

    # Streamlit UI
    st.title("Islamic Text Search Engine")

    query = st.text_input("Enter your query:")

    if query:
        def search(query, k=5):
            query_embedding = model.encode([query])
            distances, indices = index.search(query_embedding, k)
            return distances, indices

        distances, indices = search(query, k=5)

        st.write("Distances:", distances)
        st.write("Indices:", indices)

        # Fetch the corresponding texts
        def get_texts_from_indices(df, indices):
            texts = []
            for idx in indices.flatten():
                if idx < len(df):
                    texts.append(df.iloc[idx]['text_column'])
            return texts

        hadith_texts = get_texts_from_indices(hadith_df, indices)
        quran_texts = get_texts_from_indices(quran_df, indices)

        st.write("Hadith Texts:")
        st.write(hadith_texts)
        st.write("Quran Texts:")
        st.write(quran_texts)
else:
    st.write("Error loading datasets.")
