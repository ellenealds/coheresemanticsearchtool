import streamlit as st
import pandas as pd
import cohere
import time
from annoy import AnnoyIndex
co = cohere.Client('bE6Is3wvtmXyHtgnCQocDIgdH7PcYwdR21ZhnXgN') 

def embeddings(texts,sleep_time=5):
    # add a wait time to simulate a long running process
    time.sleep(sleep_time)
    response = co.embed(
        model='large',
        texts=list(texts), 
        truncate='LEFT').embeddings
    return response

df = pd.read_excel('coherefulllistoflinks.xlsx')

@st.experimental_singleton
def load_data(df,):
    df['embeddings'] = embeddings(df['text'])
    # drop rows frm text_df that havve less than 8 words
    df = df[df['text'].str.split().str.len() > 10]
    # Create the search index, pass the size of embedding
    search_index = AnnoyIndex(4096, 'angular')
    # Add all the vectors to the search index, these are stored in the dataframe 'post_members['embeddings']'
    for i, vector in enumerate(df['embeddings']):
        search_index.add_item(i, vector)
    # Build the search index
    search_index.build(10)
    #save the search index
    search_index.save('search_index.ann')
    return df, search_index

df, search_index = load_data(df)

def search(query, n_results, df, search_index, co):
    # Get the query's embedding
    query_embed = co.embed(texts=[query],
                    model="large",
                    truncate="LEFT").embeddings

    # Get the nearest neighbors and similarity score for the query and the embeddings, append it to the dataframe
    nearest_neighbors = search_index.get_nns_by_vector(query_embed[0], n_results, include_distances=True)
    # filter the dataframe to only include the nearest neighbors using the index
    df = df[df.index.isin(nearest_neighbors[0])]
    df['similarity'] = nearest_neighbors[1]
    df['nearest_neighbors'] = nearest_neighbors[0]
    df = df.sort_values(by='similarity', ascending=False)
    return df
# use threadpool executor, and concurrent modules to run the generation in parallel
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures

# define a function to generate an answer
def gen_answer(q, para): 
    response = co.generate( 
        model='command-xlarge-20221108', 
        prompt=f'Paragraph:{para}\n\nAnswer the question using this paragraph.\n\nQuestion: {q}\nAnswer:', 
        max_tokens=100, 
        temperature=0.4, 
        k=0, 
        p=0.75, 
        frequency_penalty=0, 
        presence_penalty=0, 
        stop_sequences=[], 
        return_likelihoods='NONE') 
    return response.generations[0].text

def gen_better_answer(ques, ans): 
    response = co.generate( 
        model='command-xlarge-20221108', 
        prompt=f'Answers:{ans}\n\nQuestion: {ques}\n\nGenerate a new answer that uses all the answers and makes reference to the question.', 
        max_tokens=100, 
        temperature=0.4, 
        k=0, 
        p=0.75, 
        frequency_penalty=0, 
        presence_penalty=0, 
        stop_sequences=[], 
        return_likelihoods='NONE')
        #num_generations=5) 
    return response.generations[0].text



# add a title to the app
st.title('Cohere Doc Semantic Search Tool')

# add a search bar
query = st.text_input('Search for a document')

#types = ['Blog', 'Video', 'Hackathon Examples', 'User Documentation', 'Product Documentation']


# add a card function that uses bootstrap to display the results
# I want the text to be collapsible
def card(category, text, link):
    st.markdown(f"""
    <div class="card">
        <div class="card-body">
            <h5 class="card-title">{category}</h5>
                <div class="collapse" id="collapseExample">
                    <div class="card card-body">
                    {text}
                </div>
            </div>
            <a href="{link}" class="btn btn-primary">Go to link</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# when the user clicks search, run the search function
if st.button('Search'):
    # filter the dataframe to only include the selected type
    results = search(query, 5, df, search_index, co)

    # for each row in the dataframe, generate an answer concurrently
    with ThreadPoolExecutor(max_workers=5) as executor:
        results['answer'] = list(executor.map(gen_answer, [query]*len(results), results['text']))
    #results['answer'] = results.apply(lambda x: gen_answer(query, x['text']), axis=1)
    answers = results['answer'].tolist()
    # run the function to generate a better answer
    answ = gen_better_answer(query, answers)
    #st.write(query)
    st.subheader("Cohere's answer")
    st.write(answ)
    st.subheader("Relevant documents")
    # display the results
    for i, row in results.iterrows():
        card(row['Type'], row['text'], row['link'])
       
       