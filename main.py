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
product = pd.read_csv('productinspiration.csv')

@st.experimental_singleton
def load_data(df):
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

@st.experimental_singleton
def load_data(df):
    df['embeddings'] = embeddings(df['subtitle_product_about'])
    # drop rows frm text_df that havve less than 8 words
    df = df[df['subtitle_product_about'].str.split().str.len() > 10]
    # Create the search index, pass the size of embedding
    search_index = AnnoyIndex(4096, 'angular')
    # Add all the vectors to the search index, these are stored in the dataframe 'post_members['embeddings']'
    for i, vector in enumerate(df['embeddings']):
        search_index.add_item(i, vector)
    # Build the search index
    search_index.build(10)
    #save the search index
    search_index.save('search_index_product.ann')
    return df, search_index

product, search_index_prod = load_data(product)

def search(query, n_results, df, search_index, co, type):
    with st.spinner('Cofinding relevant documents...'):

        # Get the query's embedding
        query_embed = co.embed(texts=[query],
                        model="large",
                        truncate="LEFT").embeddings
        
        # Get the nearest neighbors and similarity score for the query and the embeddings, append it to the dataframe
        if type == 'regular':
            nearest_neighbors = search_index.get_nns_by_vector(query_embed[0], n_results, include_distances=True)
        else:
            nearest_neighbors = search_index_prod.get_nns_by_vector(query_embed[0], n_results, include_distances=True)
        #nearest_neighbors = search_index.get_nns_by_vector(query_embed[0], n_results, include_distances=True)
        # filter the dataframe to only include the nearest neighbors using the index
        df = df[df.index.isin(nearest_neighbors[0])]
        df['similarity'] = nearest_neighbors[1]
        df['nearest_neighbors'] = nearest_neighbors[0]
        df = df.sort_values(by='similarity', ascending=True)
        return df

def search_project(query, n_results, df, search_index, co, filters):
    with st.spinner('Cofinding relevant documents...'):

        # Get the query's embedding
        query_embed = co.embed(texts=[query],
                        model="large",
                        truncate="LEFT").embeddings
        
        # Get the nearest neighbors and similarity score for the query and the embeddings, append it to the dataframe
        nearest_neighbors = search_index_prod.get_nns_by_vector(query_embed[0], n_results, include_distances=True)
        # filter the dataframe to only include the nearest neighbors using the index
        df = df[df.index.isin(nearest_neighbors[0])]
        df['similarity'] = nearest_neighbors[1]
        df['nearest_neighbors'] = nearest_neighbors[0]
        df = df.sort_values(by='similarity', ascending=True)
        # if filters are selected, filter the dataframe
        if filters:
            df = df[df['product'].isin(filters)]
        else:
            df = df
        return df
# use threadpool executor, and concurrent modules to run the generation in parallel
from concurrent.futures import ThreadPoolExecutor

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

# add tabs to the page
tabs = ["Cofinder", "Project Inspiration"]
choice = st.sidebar.radio("Menu", tabs)

if choice == "Cofinder":
    # add the if statements to run the search function when the user clicks the buttons
    query = st.text_input('')
    if st.button('Search'):
        results = search(query, 4, df, search_index, co, 'regular')
    # add three columns to display the buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        # add a button to search for a specific question
        if st.button('How can I build a chatbot with Cohere?'):
            query = 'How can I build a chatbot with Cohere?'
            results = search(query, 4, df, search_index, co, 'regular')
    with col2:
        if st.button('How can I build a sentiment classifier?'):
            query = 'How can I build a sentiment classifier?'
            results = search(query, 4, df, search_index, co, 'regular')
    with col3:
        if st.button('What applications can I build with Cohere endpoints?'):
            query = 'What applications can I build with Cohere endpoints?'
            results = search(query, 4, df, search_index, co, 'regular')
    # if search is empty, do nothing
    if query != '':
        # display the results
        display(query, results)
    else:  
        st.write('')

def product_ideas(queryop,prompt): 
    with st.spinner('Cofinding the best answer...'):
        response = co.generate( 
            model='e2dbd761-bcc3-4abd-9ecc-3215f049b57d-ft', 
            prompt=f"Product Type: A/B testing\nProduct Title: Text suggestions for A/B testing\nProduct Description: A/B Testing offers automated text suggestions for your headline, copy and Call To Action. Artificial Intelligence creates, combines and tests different variations of headlines, copies, calls to action and Images, to find the best fit for your audience, increasing conversions and reducing costs.   \n---\nProduct Type: Ad Generation \nProduct Title: write ads that convert using AI.\nProduct Description: Create original copy, images and creatives for Google, Facebook, Linkedin, Amazon ads and more.\n---\nProduct Type: Advertising \nProduct Title: AI-powered contextual computer vision ad solution\nProduct Description: Places ads directly into objects that are found through video and image analysis. AI that instantly transforms any image and video into shoppable moments.\n---\nProduct Type: AI Writing Assistants \nProduct Title: AI-powered autocomplete\nProduct Description: An automated writing tools is a Chrome extension that lets you automate your writing using AI.\n---\nProduct Type: Architecture \nProduct Title: AI-powered interior design\nProduct Description: A platform enables users to create fresh looks and even new features for their interior spaces.\n---\nProduct Type: Avatars  \nProduct Title: Use generative AI to create future-facing videos\nProduct Description: A generative AI converts a vision into a talking avatar.\n---\nProduct Type: Blog writing \nProduct Title: None \nProduct Description: Turn sequences of words into ready-to-send emails, messages, posts, and more while maintaining your unique tone-of-voice.\n---\nProduct Type: Book Writing \nProduct Title: AI generated book\nProduct Description: A writing application that generates stories.\n---\nProduct Type: Browser Extensions \nProduct Title: Access language models anywhere on the web\nProduct Description: Chrome Extension that lets you quickly access language models on the web. Use this extension to ask anything.\n---\nProduct Type: Bug Detection \nProduct Title: A Deep Learning Model to Detect and Fix Bugs Without Using Labelled Data\nProduct Description: Finding and fixing flaws in code necessitates not just thinking about the structure of the code but also interpreting confusing natural language cues left by software engineers in code comments, variable names, and other places.\n---\nProduct Type: Chatbots \nProduct Title: Talk to AI via WhatsApp\nProduct Description: AI Buddy let users choose different personas for the AI using WhatsApp to chat.\n---\nProduct Type: Code Explanation \nProduct Title: Using AI to make code accessible to everyone\nProduct Description: Fine-tuned model that teaches AI how to understand both code and human language.\n---\nProduct Type: Code Generation  \nProduct Title: Generate a Wordpress Plugin\nProduct Description: WordPress is web software you can use to create a beautiful website or blog, use AI to generate a wordpress plugin.\n---\nProduct Type: Code Refactoring \nProduct Title: Migrate code across languages and platforms\nProduct Description: Code translator that uses AI to translate between different languages and frameworks. \n---\nProduct Type:  Cognitive Search \nProduct Title: A search engine that provides more accurate results by understanding the user\'s intent.\nProduct Description: A cognitive search engine is a search engine that provides more accurate results by understanding the user\'s intent.\n---\nProduct Type: AI Copywriting \nProduct Title: AI-powered assistant for writing documentation.\nProduct Description: AI to Write uses AI to help you write technical documentation fast.\n---\nProduct Type: Product Management \nProduct Title: The AI product manager\nProduct Description: The AI product manager that writes your Jira tickets for you.\n---\nProduct Type: Prompt Marketplaces\nProduct Title: Use AI to find the right prompts\nProduct Description: a marketplace for buying and selling quality prompts that produce the best results, and save you money on API costs.\n---\nProduct Type: Quizes\nProduct Title: Use AI to make your next office quizes a success\nProduct Description: A tool that uses AI to help you create and run a successful office quizes.\n---\nProduct Type: Recipe Generation \nProduct Title: Build a recipe\nProduct Description: Build a delicious recipe using AI.\n---\nProduct Type: Robotics\nProduct Title: A robot that uses AI to explore its environment\nProduct Description: A robot that uses AI to explore its environment and understand language instructions.\n---\nProduct Type: Songwriting \nProduct Title: AI lyrics generator\nProduct Description: A tool that uses AI to help you write song lyrics.\n---\nProduct Type: Spreadsheets \nProduct Title: Excel integration\nProduct Description: A tool that uses AI to help you work faster and more efficiently in Microsoft Excel.\n---\nProduct Type: Tax Filling\nProduct Title: Help users with their tax returns\nProduct Description: Tax software that helps users with their tax returns by interpreting data from their bank statements into usable transaction information.\n---\nProduct Type: Gaming\nProdict Title: AI powered video game AI\nProduct Description: Video game AI powered by AI that helps players learn how to play.\n---\nProduct Type: Remote Health Monitoring\nProduct Title: Help patients monitor their health remotely\nProduct Description: Remote health monitoring tools use AI to help patients monitor their health remotely by providing information about their vitals and medications.\n---\nProduct Type: Text-to-Image\nProduct Title: AI-generated image\nProduct Description: A tool that uses AI to generate images from text.\n---\nProduct Type: Tutorials\nProduct Title: AI-generated computer programming tutorials\nProduct Description: AI-generated computer programming tutorials that use AI to help students learn how to code.\n---\nProduct Type: Voice assistants\nProduct Title: A voice assistant that uses AI to understand what you say and do\nProduct Description: A voice assistant that uses AI to understand what you say and do, and provide assistance when needed.\n---\nProduct Type: Wireframing\nProduct Title: A wireframing tool that uses AI to help you wireframe faster\nProduct Description: A wireframing tool that uses AI to help you wireframe faster.\n---\nProduct Type: Workflow Automation\nProduct Title: Use AI to automate workflows\nProduct Description: Use AI to automate workflows by automating the process of extracting data from documents, forms, and images, and inputting it into a spreadsheet.\n---\nProduct Type: Website Builders\nProduct Title: Use AI to build a website\nProduct Description: Use AI to build a website by automating the process of creating content, building a layout, and styling the site.\n---\nProduct Type: Speech Recognition\nProduct Title: Use AI to transcribe speech\nProduct Description: Use AI to transcribe speech by transcribing the audio into text.\n---\nProduct Type: Sales\nProduct Title: Use AI to increase sales\nProduct Description: Use AI to increase sales by automating the process of generating leads, converting them into customers, and managing their accounts.\n---\nProduct Type: {queryop}\n Product Title: {prompt}", 
            max_tokens=100, 
            temperature=0.7, 
            k=0, 
            p=0.75, 
            frequency_penalty=0, 
            presence_penalty=0, 
            stop_sequences=["---"], 
            return_likelihoods='NONE',
            num_generations=1) 
        return response.generations

if choice == "Project Inspiration":
    # add a search function that will update the dataframe as the user types
    # create a variable containins a unique list of subtitle
    unique = product['product'].unique()
    # pass this unqie items to a selectbox
    queryop = st.selectbox('Select a subtitle', unique)
    query = st.text_input('Add a title for your product or leave it blank to get a random idea')
    # add a button that will allow the user to search
    if st.button('Search'):
        results = product_ideas(queryop,query)
        # convert the results to a dataframe
        results = pd.DataFrame(results)
        # for each row in the table, display each row as text and add a button that will allow the user to select the row
        # show item in dataframe
        st.write(results)
        # add a button to search for similar products
    if st.button('Search for similar products'):
        st.write(search(results, 4, product, search_index_prod, co, 'not regular'))
    


    