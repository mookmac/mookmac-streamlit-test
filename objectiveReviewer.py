import os 
import tiktoken
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import PyPDF2 as ppdf2
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st  
  
# Replace this with your own OpenAI API key  
#os.environ["OPENAI_API_KEY"] = "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"  
#API_KEY = "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"  
#openai = OpenAIEmbeddings(openai_api_key=API_KEY)

def get_top_similarity_result(similarity_search_inc_score, top_n_results):
    sorted_list = sorted(similarity_search_inc_score, key=lambda x: x[1], reverse=False)         
    #print(f"Content: {sorted_list[0][0].page_content}, Metadata: {sorted_list[0][0].metadata}, Score: {sorted_list[0][1]}")
    i = 1
    all_results = []
    #print(len(sorted_list))
    while i < top_n_results and i < len(sorted_list)+1:
        all_results.append(sorted_list[i-1][0].page_content)
        #print(sorted_list[i-1][0].page_content)
        i = i + 1
    return all_results


def generate_response(input_text):
    #similar_docs = db.similarity_search(input_text)
    graphics_docs = []
    sound_docs = []
    length_docs = []
    graphics_results_with_scores = db.similarity_search_with_score("graphics")
    graphics_docs = get_top_similarity_result(graphics_results_with_scores, 4)
    sound_results_with_scores = db.similarity_search_with_score("music")
    sound_docs = get_top_similarity_result(sound_results_with_scores, 3)
    length_results_with_scores = db.similarity_search_with_score("length")
    length_docs = get_top_similarity_result(length_results_with_scores, 2)
    llm = OpenAI(temperature=0.8, openai_api_key=API_KEY, max_tokens=500, )
    #prompt    
    system_message = 'You are an AI summariser of video game reviews you generate responses based solely on the given snippets from other reviews'
    template='Using only the information provided in this prompt, write a video game review for {game} which evaluates the following points providing a balanced view: Graphics, Sound, Length. Your opinions must be based on the following information only: \
    Graphics: {graphics} \
    Sound: {sound} \
    Length: {length} \
    Do not use knowledge from any other sources. If the user does not provide information then respond with \"no input provided\" \
    Your review could follow the following structure: \
    Introduction - Give a brief overview of the game, its genre, developer. \
    Gameplay - Describe the main features of the gameplay, such as the character customization, combat system, branching storyline, and open world exploration. Mention the pros and cons of the gameplay.  \
    Graphics - Evaluate the quality of the graphics, such as the character models, environments, lighting, and animations. Mention the pros and cons of the graphics, based on the snippets. \
    Sound - Assess the quality of the sound, such as the voice acting, music, sound effects, and dialogue. Mention the pros and cons of the sound, based on the snippets. \
    Conclusion - Give your overall opinion on the game, based on your experience and the snippets. Rate the game on a scale of 1 to 10, and explain why you gave it that score. \
    Recommend the game to your audience or not, and give some reasons why. \
    Don\'t put subtitles on each section'
    prompt=PromptTemplate(input_variables=['game','graphics','sound','length'], template=template)
    prompt_query = prompt.format(game=input_text, graphics=graphics_docs, sound=sound_docs, length=length_docs)
    #print("Prompt: ", prompt_query)
    return st.info(llm(prompt_query))
  
st.set_page_config(page_title='Phenomenon Creations')
st.title('Phenomenon Creations Objective Video Game Reviewer')  
openai_api_key_override = st.sidebar.text_input('OpenAI API Key', type='password')
uploaded_files = st.sidebar.file_uploader("Upload files to use as reference data sources.",
                         type=['pdf'], accept_multiple_files=True)

text_list = []
for up_file in uploaded_files:
    pdfReader = ppdf2.PdfReader(up_file)
    for i in range(len(pdfReader.pages)):
        pageObj = pdfReader.pages[i]
        text = pageObj.extract_text()
        pageObj.clear()        
        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        docs = text_splitter.split_text(text)
        text_list.append(docs)
    print("Num pages: ", len(text_list))

  
with st.form('my_form'):
    user_input = st.text_area('Provide the name of the game you want reviewed:', 'Starfield')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key_override.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    else:
        API_KEY=openai_api_key_override
        os.environ["OPENAI_API_KEY"] = openai_api_key_override
    if submitted and openai_api_key_override.startswith('sk-') and len(text_list)>0:
        with st.spinner('Generating embeddings...'):
            embeddings = OpenAIEmbeddings(chunk_size=1000)
            db = FAISS.from_texts(docs, embeddings)
        with st.spinner('Asking AI to respond...'):
            generate_response(user_input)
