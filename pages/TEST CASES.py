from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma, Pinecone
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
import streamlit as st
import tempfile
from PIL import Image
from langchain.chains import VectorDBQA
import re
from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
import numpy as np
from langchain_community.vectorstores import FAISS
import pinecone
from langchain_community.vectorstores import Pinecone as pc
import base64
import gridfs
import openai
from langchain_community.docstore.document import Document
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
import ast
import requests
import base64
import fitz
from PIL import Image
import re
import io
from pinecone import Pinecone, ServerlessSpec
import pinecone

load_dotenv()

persist_directory = 'db'
st.title('Extract Test cases')

def extract_content(pdf_path):
    contents = []
    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        text = page.get_text("text")
        contents.append(text)
    return contents

@st.cache_resource
def loading_pdf(title):
    page_response = requests.get(title, stream=True)
    with open('input_pdf.pdf', 'wb') as f:
        f.write(page_response.content)
    loader = PyPDFLoader('input_pdf.pdf')
    data = loader.load()
    if len(data) < 90:
        contents = extract_content('input_pdf.pdf')
        contents = '\n'.join(contents)
        if contents:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=0)
            docs = [Document(page_content=x) for x in text_splitter.split_text(contents)]
            
            OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YourAPIKey')
            OPENAI_API_KEY = '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'
            llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, openai_api_key=OPENAI_API_KEY)
            chain = load_qa_chain(llm, chain_type="stuff")
            prompt = """Extract all the test cases (only Title) with the clause number from the above text. 
            Return the response as dictionary with its respective clauses. 
            Don't return any unwanted texts and headings like scope, general. 
            Final answer should be in the following format: 
            '''json {'text':[headings],'clause':[respective clauses]}'''. 
            Ensure that all strings are enclosed in double quotes. Don't return any unwanted quotes like ``` json"""
            
            response = chain.run(input_documents=docs, question=prompt,verbose=True)
            if 'json' in response:
                response = response.split('\n',1)[1]
                response = response.rsplit('\n',1)[0]
            dict_response = ast.literal_eval(response)
            
            return dict_response, docs
        
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
        texts = text_splitter.split_documents(data)
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YourAPIKey')
        OPENAI_API_KEY = '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'
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        os.environ['PINECONE_API_KEY'] = "f6365a15-09f2-43ce-8826-f4e85a824d5b"
        index_name = "langchain-demo"
        client = Pinecone(api_key="f6365a15-09f2-43ce-8826-f4e85a824d5b", environment="us")
        if index_name not in client.list_indexes().names():
            client.create_index(index_name, dimension=1536, metric='cosine',spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        index = pc.from_documents(texts, embeddings, index_name=index_name)
        index = pc.from_existing_index(index_name=index_name, embedding=embeddings)
        query = "Contents"
        similar_docs = index.similarity_search(query, k=4)
        if not similar_docs:
            count = 0
            while count<2:
                similar_docs = index.similarity_search(query, k=10)
                if similar_docs:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=0)
                    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, openai_api_key=OPENAI_API_KEY)
                    chain = load_qa_chain(llm, chain_type="stuff")
                    prompt = """Extract only the titles/headings and its title number. 
                    Don't include its sub-headings eg.4.1 or 4.1.1 so on (Only include the main titles/headings).
                    Don't include Scope,purpose definitions and other related titles. 
                    Return the response as list and don't return any unwanted texts or integers."""
                    response = chain.run(input_documents=similar_docs, question=prompt)
                    return response
                else:
                    count+=1
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=0)
            llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, openai_api_key=OPENAI_API_KEY)
            chain = load_qa_chain(llm, chain_type="stuff")
            prompt = """Extract all the titles/headings and its title number and also its sub-headings. 
            Don't include Scope,purpose definitions and other related titles. 
            Return the response as list and don't return any unwanted texts or integers."""
            response = chain.run(input_documents=similar_docs, question=prompt)
            st.write(response)
            return None, None

def extract_text(test,texts,chain):
    test_name = re.sub(r'^(-\s+)','',test)
    query = f'Extract text which is present below the heading "{test_name}" and if any encoded string is present then convert the encoded string to image and display'
    response = chain.run(input_documents=texts, question=query)
    st.write(response)

def decode_base64_to_image(base64_string):
    decoded_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(decoded_data))
    return image

def main(file_paths):
    with open(file_paths, "r") as file:
        base64_string = file.read()
    image = decode_base64_to_image(base64_string)
    st.image(image, caption='Diagram')

if __name__ == '__main__':
    if 'response' not in st.session_state:
        st.session_state.response = ''
    with st.form(key='my_form2'):
        title = st.text_input('Please add the PDF public URL', st.session_state.get('title',''), key='title')
        submit_button = st.form_submit_button(label='Submit')
        if submit_button and title:
            response, docs = loading_pdf(title)
            if not isinstance(response, list):
                st.session_state['response'] = response
                st.session_state['docs'] = docs
                test_df = pd.DataFrame(response, columns=['Tests', 'clauses'])
            elif response is None:
                client = Pinecone(api_key="f6365a15-09f2-43ce-8826-f4e85a824d5b", environment="us")
                client.delete_index("langchain-demo")
                st.stop()
    if st.session_state['response']:
        test_df = pd.DataFrame(st.session_state['response'])
        options = GridOptionsBuilder.from_dataframe(test_df, enableRowGroup=True, enableValue=True, enablePivot=True)
        options.configure_side_bar()
        options.configure_selection("single")
        selection = AgGrid(test_df, enable_enterprise_modules=True, gridOptions=options.build(), theme="streamlit", update_mode=GridUpdateMode.MODEL_CHANGED, allow_unsafe_jscode=True)
        headings = test_df['text'].to_list()
        for heading in headings:
            try:
                test_name = heading
                OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YourAPIKey')
                OPENAI_API_KEY = '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'
                llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, openai_api_key=OPENAI_API_KEY)
                chain = load_qa_chain(llm, chain_type="stuff")
                st.write(test_name)
                query = f'Extract text under the {test_name.upper()} heading including its sub classes and tables if any. Then extract the reference/other clauses mentioned in the {test_name.upper()} heading. After extracting the reference clauses extract text present under those reference clauses and combine both extracted text. This must go until all the clause text is extracted from the last extracted text (like recursive method). If any table is referred extract those also. Response should be in dictionary with key names "text" and "images" where the value for images key should be the diagram name if any diagram is mentioned in the extracted or reference heading or present under the heading. Only return the combined text, images dictionary and don’t include unwanted text like "Here is the extracted information based on your request:"'
                response = chain.run(input_documents=st.session_state['docs'], question=query)
                if "json" in response or "'''" in response.split('\n', 1)[0]:
                    response = response.split('\n', 1)[1].rsplit('\n', 1)[0]
                    response = ast.literal_eval(response)
                else:
                    response = ast.literal_eval(response)
                
                imgs = response.get('images')
                text_response = response.get('text')
                
                if imgs:
                    for file in os.listdir():
                        for img_response in imgs:
                            if re.findall(r'{}'.format(str(img_response)), file, re.IGNORECASE):
                                main(file)
                                break

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=200000, chunk_overlap=0)
                selected_case = [Document(page_content=x) for x in text_splitter.split_text(response.get('text', ''))]
                st.session_state['selected_case'] = selected_case

                test_plan_query = f'{text_response} Using the response/text before suggest the following PURPOSE, SCOPE, REFERENCES, FORMULA, TERMS, DEFINITIONS, ABBREVIATIONS, AND ACRONYMS, SAFETY REQUIREMENTS, INHERENT HAZARDS & ADEQUATE SAFEGUARDS, OPERATOR TRAINING & CERTIFICATION, PRECAUTION OF INADVERTENT CONTACT, PERSONAL PROTECTION EQUIPMENT, LASER PRECAUTIONS, TEST EQUIPMENT USED, DIAGRAM if any, TEST SETUP, TEST PROCEDURE, TABLE TEST DATA, REPORTING OF RESULTS, EVALUATION CRITERIA, SPECIFICATION REFERENCES, and DVP&R REFERENCES order wise. If any base64 strings are found then convert it to image and display under the heading "diagram". If particular detail is not available then simply return "Not enough content provided" for that particular topic/heading and don’t give any assumption response.'
                test_plan = chain.run(input_documents=st.session_state['selected_case'], question=test_plan_query)
                st.write(test_plan)
            except:
                continue
        else:
            st.stop()
