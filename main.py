import streamlit as st
import pandas as pd
import cohere
co = cohere.Client('bE6Is3wvtmXyHtgnCQocDIgdH7PcYwdR21ZhnXgN') 

def embeddings(texts,sleep_time=5):
    # add a wait time to simulate a long running process
    time.sleep(sleep_time)
    response = co.embed(
        model='large',
        texts=list(texts), 
        truncate='LEFT').embeddings
    return response

df = pd.read_excel('cohere_docs_embeddings.xlsx')

@st.cache
def load_data(df):
    df['embeddings'] = embeddings(df['text'])
    return df

df = load_data(df)

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

#df = pd.read_csv('cohere_docs_embeddings.csv')
# drop rows frm text_df that havve less than 8 words
df = df[df['text'].str.split().str.len() > 10]
from annoy import AnnoyIndex

# Create the search index, pass the size of embedding
search_index = AnnoyIndex(4096, 'angular')
# Add all the vectors to the search index, these are stored in the dataframe 'post_members['embeddings']'
for i, vector in enumerate(df['embeddings']):
    search_index.add_item(i, vector)
# Build the search index
search_index.build(10)

    
def search(query, n_results, df, search_index, co):
    # Get the query's embedding
    query_embed = co.embed(texts=[query],
                    model="large",
                    truncate="LEFT").embeddings

    # Get the nearest neighbors
    neighbors = search_index.get_nns_by_vector(query_embed[0], n_results)
    # Return the results
    return df.iloc[neighbors]

# Search for the query
results = search(search_bar, num_results, df, search_index, co)

# Display the results in a bootstrap card
for i, row in results.iterrows():
    st.write(row['text'])
