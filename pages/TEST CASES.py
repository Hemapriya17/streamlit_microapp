# from langchain_community.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader,TextLoader
# from langchain_community.vectorstores import Chroma, Pinecone
# from langchain_community.embeddings.openai import OpenAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from dotenv import load_dotenv
# from langchain_community.llms import OpenAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain_community.chat_models import ChatOpenAI
# import streamlit as st
# import tempfile
# from PIL import Image
# from langchain.chains import VectorDBQA
# import re
# # import chromadb
# # from chromadb.config import Settings
# from pymongo import MongoClient
# from langchain_community.vectorstores import MongoDBAtlasVectorSearch
# import numpy as np
# from langchain_community.vectorstores import FAISS
# import pinecone
# from langchain_community.vectorstores import Pinecone
# import base64
# import gridfs
# import openai
# from langchain_community.docstore.document import Document
# import pandas as pd
# from st_aggrid import AgGrid, GridOptionsBuilder
# from st_aggrid.shared import GridUpdateMode
# import ast
# import requests
# import base64
# import fitz
# from PIL import Image

# load_dotenv()

# persist_directory = 'db'
# st.title('Extract Test cases')

# # st.subheader('Upload your pdf')
# # uploaded_file = st.file_uploader('', type=(['pdf',"tsv","csv","txt","tab","xlsx","xls"]))
# def extract_content(blocks):
#     contents = []
#     for block in blocks:
#         if block["type"] == "Text" or block["type"] == "Table":
#             content = block["content"]
#         if block['type'] == "Figure":
#             bbox = block["bbox"]
            
#             pdf_document = fitz.open('test.pdf')
#             page_number = int(bbox['page']) - 1  # Pages are zero-indexed
#             page = pdf_document[page_number]

#             # Scale the rectangle coordinates
#             left = bbox['left'] * page.rect.width
#             top = bbox['top'] * page.rect.height
#             right = (bbox['left'] + bbox['width']) * page.rect.width
#             bottom = (bbox['top'] + bbox['height']) * page.rect.height

#             rect = fitz.Rect(left, top, right, bottom)

#             # Extract the image from the bounding box
#             img = page.get_pixmap(clip=rect)
#             img.save('test_image.png')
#             with open('test_image.png', "rb") as img_file:
#                 content = base64.b64encode(img_file.read())
#             # content = base64.b64encode(my_string)
#         contents.append(str(content))
#     return contents
# @st.cache_resource
# def loading_pdf(title):
# #     client = MongoClient('mongodb+srv://m220student:Gokulnath@cluster0.qp8h2nr.mongodb.net/')
# #     load_dotenv()
# #     db_name = "electronics"
# #     collection_name = "electronics_pdf"
# #     collection = client[db_name][collection_name]
# #     print('Collection',collection)
# #     index_name = "langchain_demo"

# #     db = client.electronics
# #     col = db.electronics_pdf
# #     # # fs = gridfs.GridFS(client)
# #     # l = ['4250.pdf','302_2_4.pdf','302_2_6.pdf'] #,'302_2_4.pdf','302_2_6.pdf','4250.pdf'
# #     # fs = gridfs.GridFS(client,collection = collection_name)
# #     OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YourAPIKey')
# #     embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# #     print('embedding working')
# #     index_name = "langchain-demo"
# #     print('pinecone')
# #     pinecone.init(
# #         api_key="d860a1e6-532d-4673-873e-f61a4099f94d", 
# #         environment="gcp-starter")
# # #     # query = "Extract all the headers which is present at the start of new line and present after a integer. Don't consider the text as heading if the integer which is present before the heading is not continuous, don't repeat the headings and return the response as list. Don't return any unwanted text and integers."
# # #     # "Extract all the headers which is present at the start of new line and present after a integer. Don't consider the text as heading if the integer which is present before the heading is not continuous. Return the response as list and don't return any unwanted texts or integers."
# # #     # print(test)
# #     print('pinecone working')
# #     index = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
# #     print(index)
# #     query = f"{title}"
# #     results = index.similarity_search_with_score(
# #         query=query,
# #         k=1
# #     )
# #     print(results)
# #     text = col.find_one({ "_id": results[0][0].metadata.get('source')})
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=200000, chunk_overlap=0)
# # # doc =  Document(page_content="text", metadata={"source": "local"})
# #     docs = [Document(page_content=x) for x in text_splitter.split_text(text.get('text'))]
# #     # texts = text_splitter.split_text(text.get('text'))
# #     # print(texts)
# #     # print(type(texts[0]))
#     url = "https://v1.api.reducto.ai/parse"

#     payload = {
#         "config": {
#             "disable_chunking": True,
#             "merge_tables": True
#         },
#         "document_url": f"{title}"
#     }
#     headers = {
#         "accept": "application/json",
#         "content-type": "application/json",
#         "authorization": "Bearer 6ed5a2d4e0b7ff14a35b4e421354f0348eb0560e9c8dfbf612d96f0383db8463d2f8a88b30fb3f72d71e3e7261daf4ea"
#     }

#     response = requests.post(url, json=payload, headers=headers)
#     print(response)
# # Parse the JSON response
#     url = 'https://utfs.io/f/ba33cfca-a907-4d7c-882d-20cf068cfcb6-98msm9.pdf'
#     page_response = requests.get(url, stream=True)
#     with open('test.pdf', 'wb') as f:
#         f.write(page_response.content)
#     data = response.json()
#     print(data)
#     for chunk in data["result"]["chunks"]:
#         contents = extract_content(chunk["blocks"])
#         contents = '\n'.join(contents)
#     # for content in contents:
#         # print(content)
#     print(contents)
#     if contents:
#         print(True)
#         print(type(contents))
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=0)
# # # doc =  Document(page_content="text", metadata={"source": "local"})
#         docs = [Document(page_content=x) for x in text_splitter.split_text(contents)]
#         OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YourAPIKey')
#         print(OPENAI_API_KEY)
#         llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, openai_api_key=OPENAI_API_KEY)
#         chain = load_qa_chain(llm, chain_type="stuff")
#         prompt = "Extract all the test cases (only Title) with the clause number from the above text. Return the response as dictionary with its respection clauses. Don't return any unwanted texts. Final answer should be in the following format: '''json {'text':[headings],'clause':[respective clauses]}'''. Ensure that all strings are enclosed in double quotes. Don't return any unwanted quotes like ``` json"
#         print('vebwbewbewb')
#         response = chain.run(input_documents=docs, question=prompt,verbose=True)
#         print(response)
#         if 'json' in response:
#             print('type',type(response))
#             response1 = response.split('\n',1)
#             print('response1',response1)
#             response = response.split('\n',1)[1]
#             print('dict_response',response)
#             response = response.rsplit('\n',1)[0]
        
#         dict_response = ast.literal_eval(response)
#         # st.write(f"Source : {results[0][0].metadata.get('source')}")
#         # st.session_state.source = results[0][0].metadata.get('source')
#         # title_prompt = "Extract the Document title from the document. Don't return any other unwanted texts or numbers. Return only the title. Remove strings like Part, Requirements and related words."
#         # title_response = chain.run(input_documents=docs, question=title_prompt)
#         # st.session_state.title_response = title_response
#         # print(title_response)
#         # st.write(f"Document Title : {title_response}")

        
# #     data = {
# #     "text": [
# #     "MARKING AND INSTRUCTIONS",
# #     "PROTECTION AGAINST ACCESS TO LIVE PARTS",
# #     "STARTING OF MOTOR-OPERATED APPLIANCES",
# #     "POWER INPUT AND CURRENT",
# #     "HEATING",
# #     "CLASSIFICATION",
# #     "LEAKAGE CURRENT AND ELECTRIC STRENGTH AT OPERATING TEMPERATURE",
# #     "TRANSIENT OVERVOLTAGES",
# #     "MOISTURE RESISTANCE",
# #     "LEAKAGE CURRENT AND ELECTRIC STRENGTH",
# #     "OVERLOAD PROTECTION OF TRANSFORMERS AND ASSOCIATED CIRCUITS",
# #     "ABNORMAL OPERATION",
# #     "STABILITY AND MECHANICAL HAZARDS",
# #     "MECHANICAL STRENGTH",
# #     "CONSTRUCTION",
# #     "INTERNAL WIRING",
# #     "COMPONENTS",
# #     "SUPPLY CONNECTION AND EXTERNAL FLEXIBLE CORDS",
# #     "TERMINALS FOR EXTERNAL CONDUCTORS",
# #     "PROVISION FOR EARTHING",
# #     "SCREWS AND CONNECTIONS",
# #     "CLEARANCES, CREEPAGE DISTANCES AND SOLID INSULATION",
# #     "RESISTANCE TO HEAT AND FIRE",
# #     "RESISTANCE TO RUSTING",
# #     "RADIATION, TOXICITY AND SIMILAR HAZARDS"
# #   ],
# #   "clause": [
# #     "7",
# #     "8",
# #     "9",
# #     "10",
# #     "11",
# #     "6",
# #     "13",
# #     "14",
# #     "15",
# #     "16",
# #     "17",
# #     "19",
# #     "20",
# #     "21",
# #     "22",
# #     "23",
# #     "24",
# #     "25",
# #     "26",
# #     "27",
# #     "28",
# #     "29",
# #     "30",
# #     "31",
# #     "32"
# #   ]
# # }
#         # test = st.radio("TEST CASES",dict_response.get('text'))
#         # test_df = pd.DataFrame(dict_response)
#         # return test_df,docs
#         return dict_response,docs
#     # return data.get('text')
#     # # test_df = pd.DataFrame(data)
#     # print(test_df.head())
#     # return test_df,'vev'
#     # test_df = pd.DataFrame(response)
#     # options = GridOptionsBuilder.from_dataframe(
#     #     test_df, enableRowGroup=True, enableValue=True, enablePivot=True
#     # )

#     # options.configure_side_bar()

#     # options.configure_selection("single")
#     # selection = AgGrid(
#     #     test_df,
#     #     enable_enterprise_modules=True,
#     #     gridOptions=options.build(),
#     #     theme="streamlit",
#     #     update_mode=GridUpdateMode.MODEL_CHANGED,
#     #     allow_unsafe_jscode=True,
#     # )
#     # docs = 'vabeb'
#     # return test_df
#     # return options,'vabawb',test_df
#     # return test_df,docs
# # for i in results:
# #     print(i[0][0].metadata)

# def extract_text(test,texts,chain):
#     print('test',test)
#     # print('texts',texts)
#     # print(chain)
#     test_name = re.sub(r'^(-\s+)','',test)
#     query = f'Extract text which is present below the heading "{test_name}" and if any encoded string is present then convert the encoded steing to image and display'
#     response = chain.run(input_documents=texts, question=query)
#     st.write(response)


# if __name__ == '__main__':
#     # test,texts,chain = loading_pdf()
#     try:
#         if 'response'not in st.session_state:

#             st.session_state.response = ''
#         # st.session_state.docs = ''
#         with st.form(key='my_form2'):
#             title = st.text_input('For which product you would like to test',st.session_state.get('title',''),key = 'title')
#             submit_button = st.form_submit_button(label='Submit')
        
#         # print(title)
#             if submit_button and title:

#                 response,docs= loading_pdf(title)
#                 print('response type',type(response))
#                 # if 'response' not in st.session_state:
#                 st.session_state['response'] = response
#                 # if 'docs' not in st.session_state:
#                 st.session_state['docs'] = docs
#                 test_df = pd.DataFrame(response,columns= ['Tests','clauses'])
#                 print(test_df.head())
#         # st.write(f"Source : {st.session_state.source}")
#         # st.write(f"Document title : {st.session_state.title_response}")
#         if st.session_state['response']:
#             test_df = pd.DataFrame(st.session_state['response'])
#             options = GridOptionsBuilder.from_dataframe(
#                 test_df, enableRowGroup=True, enableValue=True, enablePivot=True
#             )

#             options.configure_side_bar()

#             options.configure_selection("single")
#             selection = AgGrid(
#                 test_df,
#                 enable_enterprise_modules=True,
#                 gridOptions=options.build(),
#                 theme="streamlit",
#                 update_mode=GridUpdateMode.MODEL_CHANGED,
#                 allow_unsafe_jscode=True,
#             )
#             # print(selection.selected_rows[0].get('text'))

#             # test = st.radio("TEST CASES",response)
#             # test_name = re.sub(r'^(-\s+)','',test)
#             # test_name = re.sub(r'^(-\s+)','',test)
#             # print(test_name)
#             if selection.selected_rows:
#                 test_name = f'{selection.selected_rows[0].get("text")}'
#                 OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YourAPIKey')
#                 llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, openai_api_key=OPENAI_API_KEY)
#                 chain = load_qa_chain(llm, chain_type="stuff")
#                 query = f'you are a system who extract text under the {test_name} clause. If the extracted text has referrence to other clause you have to extract those clause text and if the last extracted text contains someother reference clause then you have extract those clause text also. This must go until all the clause text is extracted from the last extracted text. consider it as reference clause when it has keywords like "see","refer" etc. For some headings text will be available from the contents page but don"t include those text available in the content page. If you can"t extract the answer or if you don"t the answer, just say we don"t have answer for this  {test_name}, don"t try to make up an answer'
#                 response = chain.run(input_documents=st.session_state['docs'], question=query)
#                 st.write(response)
#                 text_splitter = RecursiveCharacterTextSplitter(chunk_size=200000, chunk_overlap=0)
# # # doc =  Document(page_content="text", metadata={"source": "local"})
#                 selected_case = [Document(page_content=x) for x in text_splitter.split_text(response)]
#                 st.session_state['selected_case'] = selected_case
#                 test_plan_query = f'{response} Using the response/text before suggest the following Objective, Apparatus Required, Manufacturers Data Required, Formula, precautions, Pre Test Condition, During Test Condition, Post Test Condition, Measured Value, Success Criteria, Procedure, Circuit Diagram if any, Tabulation and Result order wise.'
#                 test_plan = chain.run(input_documents=st.session_state['selected_case'], question=test_plan_query)
#                 print(test_plan) 
#                 st.write(test_plan)
#     except Exception as e:
#         pass
#     # extract_text(test,texts,chain)
#     # st.write(response)
#     # with st.expander('Document Similarity Search'):
#     #     # Find the relevant pages
#     #     # search = docsearch.similarity_search_with_score(prompt) 
#     #     # Write out the first 
#     #     st.write(docs[0].page_content)


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

# st.subheader('Upload your pdf')
# uploaded_file = st.file_uploader('', type=(['pdf',"tsv","csv","txt","tab","xlsx","xls"]))
def extract_content(blocks):
    contents = []
    figure = False
    content = ''
    for block in blocks:
        if block["type"] == "Text" and figure:
            fig_found = re.findall(r'\b^FIG\b|\b^fig\b',block["content"])
            # print()
            # st.write(fig_found)
            if fig_found:
                con = block["content"]
                # st.write(con)
                with open(f"{con}", "w") as output:
                    output.write(str(image_content.decode('utf8')))
                figure = False
        if block["type"] == "Text" or block["type"] == "Table" or block["type"] == "List Item" or block["type"] == "Section Header":
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
                image_content = content
            figure = True
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
    
    page_response = requests.get(title, stream=True)
    with open('input_pdf.pdf', 'wb') as f:
        f.write(page_response.content)
    loader = PyPDFLoader('input_pdf.pdf')
    data = loader.load()
    if len(data) < 30:

        # data = loader.load()
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
        # url = 'https://utfs.io/f/ba33cfca-a907-4d7c-882d-20cf068cfcb6-98msm9.pdf'
        page_response = requests.get(title, stream=True)
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
            # st.write(os.getcwd())
            # st.write(os.listdir())
            # for 
            print(True)
            print(type(contents))
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=0)
    # # doc =  Document(page_content="text", metadata={"source": "local"})
            docs = [Document(page_content=x) for x in text_splitter.split_text(contents)]
            # docs = docs[:10]
            
            OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YourAPIKey')
            OPENAI_API_KEY = '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'
            print(OPENAI_API_KEY)
            llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, openai_api_key=OPENAI_API_KEY)
            chain = load_qa_chain(llm, chain_type="stuff")
            prompt = "Extract all the test cases (only Title) with the clause number from the above text. Return the response as dictionary with its respection clauses. Don't return any unwanted texts and headings like scope, general. Final answer should be in the following format: '''json {'text':[headings],'clause':[respective clauses]}'''. Ensure that all strings are enclosed in double quotes. Don't return any unwanted quotes like ``` json"
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
    else:
        loader = PyPDFLoader('input_pdf.pdf')
        data = loader.load()
        print (f'You have {len(data)} document(s) in your data')
        print (f'There are {len(data[3].page_content)} characters in your document')
        # print(data[3].page_content)
        # merged_text = ''
        # for i in range(len(data)):
        #     merged_text += data[i].page_content + '\n'
        # print('merged_text',merged_text)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
        texts = text_splitter.split_documents(data)
        print (f'Now you have {len(texts)} documents')
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YourAPIKey')
        OPENAI_API_KEY = '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'
        print('OPENAI_API_KEY',OPENAI_API_KEY)
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        os.environ['PINECONE_API_KEY'] = "f6365a15-09f2-43ce-8826-f4e85a824d5b"
        index_name = "langchain-demo"
        client = Pinecone(api_key="f6365a15-09f2-43ce-8826-f4e85a824d5b", environment="us")
        if index_name not in client.list_indexes().names():
            print(f'Creating Index {index_name}...')
            # pinecone.create_index(index_name, dimension=1536, metric='cosine', pods=1, pod_type='p1.x2')

            client.create_index(index_name, dimension=1536, metric='cosine',spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ))

        else:
            print(f'index {index_name} already exists!')

        index = pc.from_documents(texts, embeddings, index_name=index_name)
    #     vector_store = Pinecone(index_name=index_name, embeddings=embeddings)
    
    # # Batch insert the chunks into the vector store
    #     batch_size = 100  # Define your preferred batch size
    #     for i in range(0, len(texts), batch_size):
    #         chunk_batch = texts[i:i + batch_size]
    #         vector_store.add_documents(chunk_batch)

    #     # Flush the vector store to ensure all documents are inserted
    #     vector_store.flush()
        index = pc.from_existing_index(index_name=index_name, embedding=embeddings)
        query = "Contents"
        # retriever = index.as_retriever()
        # print(retriever.get_relevant_documents(query))
        similar_docs = index.similarity_search(query, k=4)
        print(similar_docs)
        # print(type(similar_docs[0][0]))
        # print(similar_docs[0][0].page_content)
        if not similar_docs:
            count = 0
            while count<2:
                similar_docs = index.similarity_search(query, k=10)
                print(similar_docs)
                if similar_docs:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=0)
                # # doc =  Document(page_content="text", metadata={"source": "local"})
                    # docs = [Document(page_content=x) for x in text_splitter.split_text(similar_docs[0][0].page_content)]
                    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, openai_api_key=OPENAI_API_KEY)
                    chain = load_qa_chain(llm, chain_type="stuff")
                    # prompt = "Extract all the headers which is present at the start of new line and present only after a integer. Don't consider the text as heading if the integer which is present before the heading is not continuous and don't include its sub-headings. Return the response as list and don't return any unwanted texts or integers."
                    prompt = "Extract only the titles/headings and its title number. Don't include its sub-headings eg.4.1 or 4.1.1 so on (Only include the main titles/headings).Don't include Scope,purpose definitions and other related titles. Return the response as list and don't return any unwanted texts or integers."
                    response = chain.run(input_documents=similar_docs, question=prompt)
                    print(response)
                    print(count)
                    return response
                else:
                    count+=1
                    # break
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=0)
        # # doc =  Document(page_content="text", metadata={"source": "local"})
            # docs = [Document(page_content=x) for x in text_splitter.split_text(similar_docs[0][0].page_content)]
            llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, openai_api_key=OPENAI_API_KEY)
            chain = load_qa_chain(llm, chain_type="stuff")
            # prompt = "Extract all the headers which is present at the start of new line and present only after a integer. Don't consider the text as heading if the integer which is present before the heading is not continuous and don't include its sub-headings. Return the response as list and don't return any unwanted texts or integers."
            prompt = "Extract all the titles/headings and its title number and also its sub-headings. Don't include Scope,purpose definitions and other related titles. Return the response as list and don't return any unwanted texts or integers."
            response = chain.run(input_documents=similar_docs, question=prompt)
            # st.write(response)
            # print(response)
            # print(type(response))
            # response1 = list(response)
            # # print(count)
            # print(response1)
            # print(type(response1))
            return None,None
def extract_text(test,texts,chain):
    print('test',test)
    # print('texts',texts)
    # print(chain)
    test_name = re.sub(r'^(-\s+)','',test)
    query = f'Extract text which is present below the heading "{test_name}" and if any encoded string is present then convert the encoded steing to image and display'
    response = chain.run(input_documents=texts, question=query)
    st.write(response)

def decode_base64_to_image(base64_string):
    # Decode base64 string to bytes
    decoded_data = base64.b64decode(base64_string)
    print('decoded_data',decoded_data)
    # Convert bytes to image
    image = Image.open(io.BytesIO(decoded_data))
    return image

def main(file_paths):
    # for file_path in file_paths:
        # Read the base64 string from the file
        with open(file_paths, "r") as file:
            base64_string = file.read()

        # Decode base64 string to image
        image = decode_base64_to_image(base64_string)

        # Display image in Colab
        st.image(image, caption='Diagram')

if __name__ == '__main__':
    # test,texts,chain = loading_pdf()
    # try:
        if 'response'not in st.session_state:

            st.session_state.response = ''
        # st.session_state.docs = ''
        with st.form(key='my_form2'):
            title = st.text_input('Please add the PDF public URL',st.session_state.get('title',''),key = 'title')
            submit_button = st.form_submit_button(label='Submit')
        
        # print(title)
            if submit_button and title:

                response,docs= loading_pdf(title)
                print('type(response)',type(response))
                if not isinstance(response,list):
                    print('response type',type(response))
                    # if 'response' not in st.session_state:
                    st.session_state['response'] = response
                    # if 'docs' not in st.session_state:
                    st.session_state['docs'] = docs
                    test_df = pd.DataFrame(response,columns= ['Tests','clauses'])
                    print(test_df.head())
                elif response == None:
                    # st.write(response)
                    client = Pinecone(api_key="f6365a15-09f2-43ce-8826-f4e85a824d5b", environment="us")
                    client.delete_index("langchain-demo")
                    print('ggggggggggggggggggggg')
                    st.stop()
                    
        # st.write(f"Source : {st.session_state.source}")
        # st.write(f"Document title : {st.session_state.title_response}")
        if st.session_state['response']:
            test_df = pd.DataFrame(st.session_state['response'])
            # st.write(test_df.head())
            # st.write(test_df.columns)
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
            headings = test_df['text'].to_list()
            for heading in headings:
                # options = GridOptionsBuilder.from_dataframe(
                #     test_df, enableRowGroup=True, enableValue=True, enablePivot=True
                # )

                # options.configure_side_bar()

                # options.configure_selection("single")
                # print('gggggggggggggggggggggggg')
                # selection = AgGrid(
                #     test_df,
                #     enable_enterprise_modules=True,
                #     gridOptions=options.build(),
                #     theme="streamlit",
                #     update_mode=GridUpdateMode.MODEL_CHANGED,
                #     allow_unsafe_jscode=True,
                # )
                # print(selection.selected_rows[0].get('text'))

            # # test = st.radio("TEST CASES",response)
            # # test_name = re.sub(r'^(-\s+)','',test)
            # # test_name = re.sub(r'^(-\s+)','',test)
            # # print(test_name)
            # # if selection is not None:
            # if selection is not None and hasattr(selection, 'selected_rows') and selection.selected_rows:
            #     print('hhhhhhhhhhhhhhhhhh')
            #     test_name = f'{selection.selected_rows[0].get("text")}'
                test_name = heading
                OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YourAPIKey')
                OPENAI_API_KEY = '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'
                llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, openai_api_key=OPENAI_API_KEY)
                chain = load_qa_chain(llm, chain_type="stuff")
                st.write(test_name)
                # query = f'you are a system who extract text under the {test_name.upper()} heading including its sub classes. If the extracted text has referrence to other clause you need to extract those heading/clause text and if the last extracted text contains someother reference clause then you have extract those clause text also. This must go until all the clause text is extracted from the last extracted text. consider it as reference clause when it has keywords like "see","refer" etc. For some headings text will be available from the contents page but don"t include those text available in the content page. If you can"t extract the answer or if you don"t the answer, just say we don"t have answer for this  {test_name}, don"t try to make up an answer. Return value should be in tuple where first value should be the text and the second value in the tuple should be the figure number if any figure number is mentioned in the extracted text/ present under the heading.'
                query = f'Extract text under the {test_name.upper()} heading including its sub classes and tables if any. If the extracted text has referrence to other clause you need to extract those heading/clause text .This must go until all the clause text is extracted from the last extracted text. consider it as reference clause when it has keywords like "see","refer" etc. If any table is refered extract those also.Response should be in tuple where first value should be the text and the second value in the tuple should be the figure name if any figure name is mentioned in the extracted text/ present under the heading. only return the tuple and don"t include unwanted text'
                response = chain.run(input_documents=st.session_state['docs'], question=query)
                # st.write(response)
                # response = 'FIG. 104'
                # st.write(response.rsplit(',',1)[-1])
                img_response=response.rsplit(',',1)[-1]
                img_response = img_response.replace("'",'')
                img_response = img_response.replace('"','').strip()
                paren = True
                while paren:
                    img_response_1 = img_response
                    img_response = re.sub(r'\)','',img_response)
                    if img_response == img_response_1:
                        paren = False
                    # paren = False
                # img_response = img_response[0:len(img_response)-1]
                # print(img_response,img_response)
                # st.write(img_response)
                # response = ast.literal_eval(response)
                for file in os.listdir():
                    # print('file',file)
                    # st.write(file)
                    # st.write(response.rsplit(',')[1])
                    if img_response:
                        # st.write(img_response)
                        print('img_response',img_response)
                        file_found = re.findall(r'{}'.format(str(img_response)),file,re.IGNORECASE)
                        print('file_found',file_found)
                        # st.write(file_found)
                        if file_found:
                            # print('brebrbn')
                            # f = open(f"{file}", "r")
                            # image_base64 = f.read()
                            # st.write(f.read())
                            main(file)
                    
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=200000, chunk_overlap=0)
                # # doc =  Document(page_content="text", metadata={"source": "local"})
                selected_case = [Document(page_content=x) for x in text_splitter.split_text(response)]
                st.session_state['selected_case'] = selected_case
                st.write(response)
                test_plan_query = f'{response} Using the response/text before suggest the following Objective, Apparatus Required, Manufacturers Data Required, Formula, precautions, Pre Test Condition, During Test Condition, Post Test Condition, Measured Value, Success Criteria, Procedure, Circuit Diagram if any, Tabulation and Result order wise. If any base64 string are found then convert it to image and display under the heading "diagram". If enough contexts/text is not available then simply return "Not enough content provided" don"t give any assumption response.'
                test_plan = chain.run(input_documents=st.session_state['selected_case'], question=test_plan_query)
                print(test_plan) 
                st.write(test_plan)
            else:
                st.stop()
    # except Exception as e:
    #     pass
    # extract_text(test,texts,chain)
    # st.write(response)
    # with st.expander('Document Similarity Search'):
    #     # Find the relevant pages
    #     # search = docsearch.similarity_search_with_score(prompt) 
    #     # Write out the first 
    #     st.write(docs[0].page_content)











