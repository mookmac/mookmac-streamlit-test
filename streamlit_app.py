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
os.environ["OPENAI_API_KEY"] = "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"  
API_KEY = "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"  
openai = OpenAIEmbeddings(openai_api_key=API_KEY)
   
def generate_response(input_text):
    similar_docs = db.similarity_search(input_text)
    results_with_scores = db.similarity_search_with_score(input_text)
    for doc, score in results_with_scores:
        print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")

    llm = OpenAI(temperature=0.7, openai_api_key=API_KEY, max_tokens=150)
    #prompt
    #template='You are an expert in completing the Disability Living Allowance form. Relevant published guidance includes: {similar}. Explain the following to me in simple terms: {question}'
    template='Using this information only: {similar}. Explain the following to me in simple terms: {question}'
    prompt=PromptTemplate(input_variables=['similar','question'], template=template)
    prompt_query = prompt.format(similar=similar_docs, question=input_text)
    print("PROMPT: ", prompt_query)
    return st.info(llm(prompt_query))
  
st.set_page_config(page_title='Phenomenon Creations')
st.title('Phenomenon Creations Complex Form Completion Assistant')  
openai_api_key_override = st.sidebar.text_input('OpenAI API Key', type='password')
uploaded_files = st.sidebar.file_uploader("Upload files to use as reference data sources.",
                         type=['pdf'], accept_multiple_files=True)

text_list = []
for up_file in uploaded_files:
    pdfReader = ppdf2.PdfReader(up_file)
    for i in range(len(pdfReader.pages)):
        pageObj = pdfReader.pages[i]
        text = pageObj.extract_text()
        print("Page ", i, ": ", text)
        pageObj.clear()
        text_list.append(text)
    print("Num pages: ", len(text_list))



  
with st.form('my_form'):
    user_input = st.text_area('Enter your message:', 'What does question 2 mean?')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key_override.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key_override.startswith('sk-') and len(text_list)>0:
        with st.spinner('Generating embeddings...'):
            embeddings = OpenAIEmbeddings(chunk_size=1000)
            db = FAISS.from_texts(text_list, embeddings)
        with st.spinner('Asking AI to respond...'):
            generate_response(user_input)
