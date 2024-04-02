from langchain_community.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader,TextLoader
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
# import chromadb
# from chromadb.config import Settings
from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
import numpy as np
from langchain_community.vectorstores import FAISS
import pinecone
from langchain_community.vectorstores import Pinecone
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

load_dotenv()

persist_directory = 'db'
st.title('Extract Test cases')

# st.subheader('Upload your pdf')
# uploaded_file = st.file_uploader('', type=(['pdf',"tsv","csv","txt","tab","xlsx","xls"]))
def extract_content(blocks):
    contents = []
    for block in blocks:
        if block["type"] == "Text" or block["type"] == "Table":
            content = block["content"]
        if block['type'] == "Figure":
            bbox = block["bbox"]
            
            pdf_document = fitz.open('test.pdf')
            page_number = int(bbox['page']) - 1  # Pages are zero-indexed
            page = pdf_document[page_number]

            # Scale the rectangle coordinates
            left = bbox['left'] * page.rect.width
            top = bbox['top'] * page.rect.height
            right = (bbox['left'] + bbox['width']) * page.rect.width
            bottom = (bbox['top'] + bbox['height']) * page.rect.height

            rect = fitz.Rect(left, top, right, bottom)

            # Extract the image from the bounding box
            img = page.get_pixmap(clip=rect)
            img.save('test_image.png')
            with open('test_image.png', "rb") as img_file:
                content = base64.b64encode(img_file.read())
            # content = base64.b64encode(my_string)
        contents.append(str(content))
    return contents
@st.cache_resource
def loading_pdf(title):
#     client = MongoClient('mongodb+srv://m220student:Gokulnath@cluster0.qp8h2nr.mongodb.net/')
#     load_dotenv()
#     db_name = "electronics"
#     collection_name = "electronics_pdf"
#     collection = client[db_name][collection_name]
#     print('Collection',collection)
#     index_name = "langchain_demo"

#     db = client.electronics
#     col = db.electronics_pdf
#     # # fs = gridfs.GridFS(client)
#     # l = ['4250.pdf','302_2_4.pdf','302_2_6.pdf'] #,'302_2_4.pdf','302_2_6.pdf','4250.pdf'
#     # fs = gridfs.GridFS(client,collection = collection_name)
#     OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YourAPIKey')
#     embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
#     print('embedding working')
#     index_name = "langchain-demo"
#     print('pinecone')
#     pinecone.init(
#         api_key="d860a1e6-532d-4673-873e-f61a4099f94d", 
#         environment="gcp-starter")
# #     # query = "Extract all the headers which is present at the start of new line and present after a integer. Don't consider the text as heading if the integer which is present before the heading is not continuous, don't repeat the headings and return the response as list. Don't return any unwanted text and integers."
# #     # "Extract all the headers which is present at the start of new line and present after a integer. Don't consider the text as heading if the integer which is present before the heading is not continuous. Return the response as list and don't return any unwanted texts or integers."
# #     # print(test)
#     print('pinecone working')
#     index = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
#     print(index)
#     query = f"{title}"
#     results = index.similarity_search_with_score(
#         query=query,
#         k=1
#     )
#     print(results)
#     text = col.find_one({ "_id": results[0][0].metadata.get('source')})
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=200000, chunk_overlap=0)
# # doc =  Document(page_content="text", metadata={"source": "local"})
#     docs = [Document(page_content=x) for x in text_splitter.split_text(text.get('text'))]
#     # texts = text_splitter.split_text(text.get('text'))
#     # print(texts)
#     # print(type(texts[0]))
    url = "https://v1.api.reducto.ai/parse"

    payload = {
        "config": {
            "disable_chunking": True,
            "merge_tables": True
        },
        "document_url": f"{title}"
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": "Bearer 6ed5a2d4e0b7ff14a35b4e421354f0348eb0560e9c8dfbf612d96f0383db8463d2f8a88b30fb3f72d71e3e7261daf4ea"
    }

    response = requests.post(url, json=payload, headers=headers)
    print(response)
# Parse the JSON response
    url = 'https://utfs.io/f/ba33cfca-a907-4d7c-882d-20cf068cfcb6-98msm9.pdf'
    page_response = requests.get(url, stream=True)
    with open('test.pdf', 'wb') as f:
        f.write(page_response.content)
    data = response.json()
    print(data)
    for chunk in data["result"]["chunks"]:
        contents = extract_content(chunk["blocks"])
        contents = '\n'.join(contents)
    # for content in contents:
        # print(content)
    print(contents)
    if contents:
        print(True)
        print(type(contents))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=0)
# # doc =  Document(page_content="text", metadata={"source": "local"})
        docs = [Document(page_content=x) for x in text_splitter.split_text(contents)]
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YourAPIKey')
        print(OPENAI_API_KEY)
        llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, openai_api_key=OPENAI_API_KEY)
        chain = load_qa_chain(llm, chain_type="stuff")
        prompt = "Extract all the test cases (only Title) with the clause number from the above text. Return the response as dictionary with its respection clauses. Don't return any unwanted texts. Final answer should be in the following format: '''json {'text':[headings],'clause':[respective clauses]}'''. Ensure that all strings are enclosed in double quotes. Don't return any unwanted quotes like ``` json"
        print('vebwbewbewb')
        response = chain.run(input_documents=docs, question=prompt,verbose=True)
        print(response)
        if 'json' in response:
            print('type',type(response))
            response1 = response.split('\n',1)
            print('response1',response1)
            response = response.split('\n',1)[1]
            print('dict_response',response)
            response = response.rsplit('\n',1)[0]
        
        dict_response = ast.literal_eval(response)
        # st.write(f"Source : {results[0][0].metadata.get('source')}")
        # st.session_state.source = results[0][0].metadata.get('source')
        # title_prompt = "Extract the Document title from the document. Don't return any other unwanted texts or numbers. Return only the title. Remove strings like Part, Requirements and related words."
        # title_response = chain.run(input_documents=docs, question=title_prompt)
        # st.session_state.title_response = title_response
        # print(title_response)
        # st.write(f"Document Title : {title_response}")

        
#     data = {
#     "text": [
#     "MARKING AND INSTRUCTIONS",
#     "PROTECTION AGAINST ACCESS TO LIVE PARTS",
#     "STARTING OF MOTOR-OPERATED APPLIANCES",
#     "POWER INPUT AND CURRENT",
#     "HEATING",
#     "CLASSIFICATION",
#     "LEAKAGE CURRENT AND ELECTRIC STRENGTH AT OPERATING TEMPERATURE",
#     "TRANSIENT OVERVOLTAGES",
#     "MOISTURE RESISTANCE",
#     "LEAKAGE CURRENT AND ELECTRIC STRENGTH",
#     "OVERLOAD PROTECTION OF TRANSFORMERS AND ASSOCIATED CIRCUITS",
#     "ABNORMAL OPERATION",
#     "STABILITY AND MECHANICAL HAZARDS",
#     "MECHANICAL STRENGTH",
#     "CONSTRUCTION",
#     "INTERNAL WIRING",
#     "COMPONENTS",
#     "SUPPLY CONNECTION AND EXTERNAL FLEXIBLE CORDS",
#     "TERMINALS FOR EXTERNAL CONDUCTORS",
#     "PROVISION FOR EARTHING",
#     "SCREWS AND CONNECTIONS",
#     "CLEARANCES, CREEPAGE DISTANCES AND SOLID INSULATION",
#     "RESISTANCE TO HEAT AND FIRE",
#     "RESISTANCE TO RUSTING",
#     "RADIATION, TOXICITY AND SIMILAR HAZARDS"
#   ],
#   "clause": [
#     "7",
#     "8",
#     "9",
#     "10",
#     "11",
#     "6",
#     "13",
#     "14",
#     "15",
#     "16",
#     "17",
#     "19",
#     "20",
#     "21",
#     "22",
#     "23",
#     "24",
#     "25",
#     "26",
#     "27",
#     "28",
#     "29",
#     "30",
#     "31",
#     "32"
#   ]
# }
        # test = st.radio("TEST CASES",dict_response.get('text'))
        # test_df = pd.DataFrame(dict_response)
        # return test_df,docs
        return dict_response,docs
    # return data.get('text')
    # # test_df = pd.DataFrame(data)
    # print(test_df.head())
    # return test_df,'vev'
    # test_df = pd.DataFrame(response)
    # options = GridOptionsBuilder.from_dataframe(
    #     test_df, enableRowGroup=True, enableValue=True, enablePivot=True
    # )

    # options.configure_side_bar()

    # options.configure_selection("single")
    # selection = AgGrid(
    #     test_df,
    #     enable_enterprise_modules=True,
    #     gridOptions=options.build(),
    #     theme="streamlit",
    #     update_mode=GridUpdateMode.MODEL_CHANGED,
    #     allow_unsafe_jscode=True,
    # )
    # docs = 'vabeb'
    # return test_df
    # return options,'vabawb',test_df
    # return test_df,docs
# for i in results:
#     print(i[0][0].metadata)

def extract_text(test,texts,chain):
    print('test',test)
    # print('texts',texts)
    # print(chain)
    test_name = re.sub(r'^(-\s+)','',test)
    query = f'Extract text which is present below the heading "{test_name}" and if any encoded string is present then convert the encoded steing to image and display'
    response = chain.run(input_documents=texts, question=query)
    st.write(response)


if __name__ == '__main__':
    # test,texts,chain = loading_pdf()
    # try:
        if 'response'not in st.session_state:

            st.session_state.response = ''
        # st.session_state.docs = ''
        with st.form(key='my_form2'):
            title = st.text_input('For which product you would like to test',st.session_state.get('title',''),key = 'title')
            submit_button = st.form_submit_button(label='Submit')
        
        # print(title)
            if submit_button and title:

                response,docs= loading_pdf(title)
                print('response type',type(response))
                # if 'response' not in st.session_state:
                st.session_state['response'] = response
                # if 'docs' not in st.session_state:
                st.session_state['docs'] = docs
                test_df = pd.DataFrame(response,columns= ['Tests','clauses'])
                print(test_df.head())
        # st.write(f"Source : {st.session_state.source}")
        # st.write(f"Document title : {st.session_state.title_response}")
        if st.session_state['response']:
            test_df = pd.DataFrame(st.session_state['response'])
            options = GridOptionsBuilder.from_dataframe(
                test_df, enableRowGroup=True, enableValue=True, enablePivot=True
            )

            options.configure_side_bar()

            options.configure_selection("single")
            print('gggggggggggggggggggggggg')
            selection = AgGrid(
                test_df,
                enable_enterprise_modules=True,
                gridOptions=options.build(),
                theme="streamlit",
                update_mode=GridUpdateMode.MODEL_CHANGED,
                allow_unsafe_jscode=True,
            )
            # print(selection.selected_rows[0].get('text'))

            # test = st.radio("TEST CASES",response)
            # test_name = re.sub(r'^(-\s+)','',test)
            # test_name = re.sub(r'^(-\s+)','',test)
            # print(test_name)
            if selection.selected_rows:
                print('hhhhhhhhhhhhhhhhhh')
                test_name = f'{selection.selected_rows[0].get("text")}'
                OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YourAPIKey')
                llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, openai_api_key=OPENAI_API_KEY)
                chain = load_qa_chain(llm, chain_type="stuff")
                query = f'you are a system who extract text under the {test_name} clause. If the extracted text has referrence to other clause you have to extract those clause text and if the last extracted text contains someother reference clause then you have extract those clause text also. This must go until all the clause text is extracted from the last extracted text. consider it as reference clause when it has keywords like "see","refer" etc. For some headings text will be available from the contents page but don"t include those text available in the content page. If you can"t extract the answer or if you don"t the answer, just say we don"t have answer for this  {test_name}, don"t try to make up an answer'
                response = chain.run(input_documents=st.session_state['docs'], question=query)
                st.write(response)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=200000, chunk_overlap=0)
# # doc =  Document(page_content="text", metadata={"source": "local"})
                selected_case = [Document(page_content=x) for x in text_splitter.split_text(response)]
                st.session_state['selected_case'] = selected_case
                test_plan_query = f'{response} Using the response/text before suggest the following Objective, Apparatus Required, Manufacturers Data Required, Formula, precautions, Pre Test Condition, During Test Condition, Post Test Condition, Measured Value, Success Criteria, Procedure, Circuit Diagram if any, Tabulation and Result order wise.'
                test_plan = chain.run(input_documents=st.session_state['selected_case'], question=test_plan_query)
                print(test_plan) 
                st.write(test_plan)
    # except Exception as e:
    #     pass
    # extract_text(test,texts,chain)
    # st.write(response)
    # with st.expander('Document Similarity Search'):
    #     # Find the relevant pages
    #     # search = docsearch.similarity_search_with_score(prompt) 
    #     # Write out the first 
    #     st.write(docs[0].page_content)







