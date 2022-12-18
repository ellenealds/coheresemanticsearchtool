import streamlit as st
import pandas as pd

@st.cache
def load_data():
    df = pd.read_csv('cohere_docs_embeddings.csv')
    return df

df = load_data()

st.title('Cohere Doc Semantic Search Tool')

# add a search bar
search_bar = st.text_input('Search for a document')

# add a select box for the search type

search_type = st.selectbox('Search Type', ('Exact', 'Fuzzy'))

# add a slider for the number of results to return

num_results = st.slider('Number of Results', 1, 10, 5)

# add a button to trigger the search

if st.button('Search'):
    # do the search
    if search_type == 'Exact':
        results = df[df['text'].str.contains(search_bar)]
    else:
        results = df[df['text'].str.contains(search_bar, case=False)]
    # display the results
    st.write(results.head(num_results))

