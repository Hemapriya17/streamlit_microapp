# import os
# os.system("python -m plantuml minmap.txt")
# import streamlit as st
# st.image('minmap.png', caption='Sunrise by the mountains')

import os
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import json
import pandas as pd
import ast
import time
import openai

import base64
from PIL import Image
import base64
import requests
import io
import re
import os
import plantuml

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])  # this is also the default, it can be omitted)

def generate_mindmap_name():
    
        # print('vezbezb')
        mindmap_name = st.text_input('NAME',placeholder = 'Enter the process to generate PlantUML')
        tmp_button = st.button(label='Submit')
        print(tmp_button,mindmap_name)
        if tmp_button and mindmap_name:
            prompt = f'generate Mermaid syntax for {mindmap_name} with @startmindmap at the start and @endmindmap at the end. Don"t return any numbers or any unwanted text, description of the response. Don"t return flowchart diagram.'
            messages=[{"role": "user", "content": prompt}]
            start_time = time.time()
            completion = client.chat.completions.create(model="gpt-4-1106-preview",messages=messages,temperature = 0)
            # print('bezbezb')
            end_time = time.time()
            time_lapsed = end_time - start_time
            # st.write(start_time,end_time)
            st.write(f'{round(time_lapsed, 2)} secs')
            response = completion.choices[0].message.content
            # print(response)
            text_file = open("mindmap.txt", "w")
            text_file.write(response)
            text_file.close()
            os.system("python -m plantuml mindmap.txt -o pages")
            # st.write(os.listdir())
            os.system("cat -n .streamlit")
            # st.write(os.getcwd())
            os.chdir('pages')
            # st.write(os.listdir())
            os.chdir('..')
            # st.write(os.listdir())
            st.image('pages/mindmap.png')
            response1 = response.split('\n',1)
            print('response1',response1)
            response = response.split('\n',1)[1]
            print('dict_response',response)
            graph = response.rsplit('\n',1)[0]
            # st.write(response)
            graphbytes = graph.encode("ascii")
            base64_bytes = base64.b64encode(graphbytes)
            base64_string = base64_bytes.decode("ascii")

            # print(base64_string)

            img = Image.open(io.BytesIO(requests.get('https://mermaid.ink/img/' + base64_string).content))
            st.image(img, caption='Processed Image', use_column_width=True)

            st.markdown(f"```mermaid\n{response}\n```", unsafe_allow_html=True)

if __name__ == '__main__':
    generate_mindmap_name()
