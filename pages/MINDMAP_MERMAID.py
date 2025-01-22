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

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def generate_sequencediagram():
    
        # print('vezbezb')
        name = st.text_input('NAME',placeholder = 'Enter the process to generate Mindmap')
        tmp_button = st.button(label='Submit')
        print(tmp_button,name)
        if tmp_button and name:
            prompt = f'Generate mindmap diagram (mermaid code) for {name} process. Don"t return any numbers or any unwanted text, description of the response. Don"t return flowchart diagram.'
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
            # response1 = response.split('\n',1)
            # print('response1',response1)
            response = response.split('\n',1)[1]
            print('dict_response',response)
            graph = response.rsplit('\n',1)[0]
            # st.write(response)
            graphbytes = graph.encode("ascii")
            base64_bytes = base64.b64encode(graphbytes)
            base64_string = base64_bytes.decode("ascii")

            # Fetch the image from the URL
            response = requests.get('https://mermaid.ink/img/' + base64_string)
            
            # Check if the response is successful and contains image data
            if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
                img = Image.open(io.BytesIO(response.content))
                st.image(img, caption='Processed Image', use_column_width=True)
            else:
                st.error("Failed to retrieve a valid image. Please check the input or try again.")

if __name__ == '__main__':
    generate_sequencediagram()
