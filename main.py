import streamlit as st
import pandas as pd
import cohere
import time
from annoy import AnnoyIndex
from concurrent.futures import ThreadPoolExecutor
 

# COHERE_API_KEY is stored in streamlit secrets
COHERE_API_KEY = st.secrets["COHERE_API_KEY"]
co = cohere .Client(COHERE_API_KEY)

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
def load_data(df):
    df['embeddings'] = embeddings(df['text'])
    # drop rows frm text_df that havve less than 8 words
    df = df[df['text'].str.split().str.len() > 10]
    # Create the search index, pass the size of embedding
    search_index = AnnoyIndex(4096, 'angular')
    # Add all the vectors to the search index, these are stored in the dataframe
    for i, vector in enumerate(df['embeddings']):
        search_index.add_item(i, vector)
    # Build the search index
    search_index.build(10)
    #save the search index
    search_index.save('search_index.ann')
    return df, search_index

df, search_index = load_data(df)


def search(query, n_results, df, search_index, co, type):
    with st.spinner('Cofinding relevant documents...'):

        # Get the query's embedding
        query_embed = co.embed(texts=[query],
                        model="large",
                        truncate="LEFT").embeddings
        # Get the nearest neighbors and similarity score for the query and the embeddings, append it to the dataframe
        if type == 'regular':
            nearest_neighbors = search_index.get_nns_by_vector(query_embed[0], n_results, include_distances=True)
        #nearest_neighbors = search_index.get_nns_by_vector(query_embed[0], n_results, include_distances=True)
        # filter the dataframe to only include the nearest neighbors using the index
        df = df[df.index.isin(nearest_neighbors[0])]
        df['similarity'] = nearest_neighbors[1]
        df['nearest_neighbors'] = nearest_neighbors[0]
        df = df.sort_values(by='similarity', ascending=True)
        return df


# define a function to generate an answer
def gen_answer(q, para):  
    if len(para.split()) > 1800:
            # truncate the string
            para = para[:1800]
    else:
        para = para
    with st.spinner('Cofinding answers...'):
        response = co.generate( 
            model='command-xlarge-20221108', 
            #if para contains more than 1900 tokens truncate it
            prompt=f'Paragraph:{para}\n\nAnswer the query using this paragraph.\n\Query: {q}\nAnswer:', 
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
    with st.spinner('Cofinding the best answer...'):
        response = co.generate( 
            model='command-xlarge-20221108', 
            prompt=f'Answers:{ans}\n\Query: {ques}\n\nGenerate a new answer that uses the best answers and makes reference to the query.', 
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

def display(query, results):
    # for each row in the dataframe, generate an answer concurrently
    with ThreadPoolExecutor(max_workers=5) as executor:
        results['answer'] = list(executor.map(gen_answer, [query]*len(results), results['text']))
    #results['answer'] = results.apply(lambda x: gen_answer(query, x['text']), axis=1)
    answers = results['answer'].tolist()
    # if the combination of answers contains more than 2000 tokens, then truncate the list
    if len(' '.join(answers).split()) > 1800:
        answers = answers[:1800]
    # run the function to generate a better answer
    answ = gen_better_answer(query, answers)
    #st.write(query)
    st.subheader(query)
    st.write(answ)
    # add a spacer
    st.write('')
    st.write('')
    st.subheader("Relevant documents")
    # display the results
    for i, row in results.iterrows():
        #there are 2 columns and 3 rows
        
        st.markdown(f'**{row["Type"]}**')
        st.markdown(f'**{row["Category"]}**')
        st.markdown(f'{row["title"]}')
        # display the url as a hyperlink and add a button to open the url in a new tab
        st.write(row['answer'])
        # collapse the text
        with st.expander('View page'):
            st.markdown(f'[{row["link"]}]({row["link"]})')
            st.markdown(f'<a href="{row["link"]}" target="_blank">Open in new tab</a>', unsafe_allow_html=True)
            # add an iframe for the link
            # if the link is a youtube video, then add the media player
            if 'youtube' in row['link']:
                # display the media player
                st.video(row["link"])
            else:
                # display the iframe
                st.write(f'<iframe src="{row["link"]}" width="700" height="1000"></iframe>', unsafe_allow_html=True)
            st.write('')      
def display_product(df):
    # display the dataframe as a table and on;y inclide the columns subtitle, url, and about
    st.table(df[['product','subtitle', 'about']])



# add an image to the top of the page, the image is 'beach.png'
st.image('beach.png', width=700)
# add a subtitle
st.subheader("A semantic search tool built for the Cohere community")

# add a smaller text
# add a collapsible section to display the text and label it About
with st.expander('About'):
    st.write("This tool uses the Cohere API to search through the Cohere knowledge base and generate answers to questions. It uses the Cohere embed endpoint to find relevant documents, and the Cohere generate endpoint to generate answers to questions.")
    st.write('Select a question from the examples or ask your own using the search function.')

# add the if statements to run the search function when the user clicks the buttons
query = st.text_input('')
if st.button('Search'):
    results = search(query, 4, df, search_index, co, 'regular')
# add three columns to display the buttons
col1, col2, col3, col4 = st.columns(4)
with col1:
    # add a button to search for a specific question
    if st.button('How can I build a chatbot with Cohere?'):
        query = 'How can I build a chatbot with Cohere?'
        results = search(query, 4, df, search_index, co, 'regular')
with col2:
    if st.button('What can I build with the sentiment classifier?'):
        query = 'What can I build a sentiment classifier?'
        results = search(query, 4, df, search_index, co, 'regular')
with col3:
    if st.button('How can I use Cohere to create value for my business?'):
        query = 'How can I use Cohere to create value for my business?'
        results = search(query, 4, df, search_index, co, 'regular')

with col4:
    if st.button('How can I use Cohere to make efficiencies?'):
        query = 'How can I use Cohere to make efficiencies?'
        results = search(query, 4, df, search_index, co, 'regular')
# if search is empty, do nothing
if query != '':
    # display the results
    display(query, results)
else:  
    st.write('')

