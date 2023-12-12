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
import nltk
from nltk import word_tokenize, pos_tag

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])  # this is also the default, it can be omitted)

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def generate_processmap():
    
        # print('vezbezb')
        processmap_name = st.text_input('Please mention with both action and object(Eg Cleaning a coffee maker, making coffee using coffee maker). For objects like washing machine please mention it by adding a "-" like washing-machine',' ',placeholder = 'Enter the name for which you would like to generate process map')
        tmp_button = st.button(label='Submit')
        print(tmp_button,processmap_name)
        if tmp_button and processmap_name:
            words = word_tokenize(processmap_name)

    # Perform part-of-speech tagging
            pos_tags = pos_tag(words)

            # Extract action and object words based on POS tags
            # st.write(pos_tags)
            actions = [word for word, pos in pos_tags if pos.startswith('VB')]  # VB stands for verbs
            objects = [word for word, pos in pos_tags if pos.startswith('NN')]  # NN stands for nouns
            # st.write(actions,objects)
            if actions and objects:
            # action_prompt = f'Return action and object text from {processmap_name} text.return the response as json.Don"t return any unwanted text or assumption.'
                # messages=[{"role": "user", "content": action_prompt}]
                # completion = client.chat.completions.create(model="gpt-4-1106-preview",messages=messages,temperature = 0)
                # # print('bezbezb')
                # response = completion.choices[0].message.content
                # print('ggggggg',response)
                prompt = f'Generate graph map (mermaid code) for {processmap_name} including output for each step . The output should be in between vertical slash and the input should be in between square brackets. The process should be ending with only a input and don"t include yes or no outputs or any questioning outputs or inputs. The graph map should be in sequence. Don"t return any numbers or any unwanted text, description of the response. Don"t return flowchart diagram.'

                #Generate steps involved in {processmap_name} (steps should only be in 2 -3 words)and mention the output for each step in 2-3 words. Return each step in a list and don"t return any numbers or other unwanted texts output string
                messages=[{"role": "user", "content": prompt}]
                completion = client.chat.completions.create(model="gpt-4-1106-preview",messages=messages,temperature = 0)
                # print('bezbezb')
                response = completion.choices[0].message.content
                print(response)
                # response = response.split('\n')
                # st.write(response)
                response1 = response.split('\n',1)
                print('response1',response1)
                response = response.split('\n',1)[1]
                print('dict_response',response)
                graph = response.rsplit('\n',1)[0]
                graph = graph.replace('/','')
                # st.write(response)
                
                graphbytes = graph.encode("ascii")
                base64_bytes = base64.b64encode(graphbytes)
                base64_string = base64_bytes.decode("ascii")

                # print(base64_string)
                print(requests.get('https://mermaid.ink/img/' + base64_string))
                img = Image.open(io.BytesIO(requests.get('https://mermaid.ink/img/' + base64_string).content))
                st.image(img, caption='Processed Image', use_column_width=True)
            else:
                if not actions and not objects:
                    st.write('Please try again with by specifying both action and object')
                elif not actions:
                    st.write(f'You have specified only the object {objects}. Please mention some action like cleaning, Building etc. to generate a process map.')
                elif not objects:
                    st.write(f'You have specified only the action {actions}. Please mention some objects like coffee maker, washing machine to generate a process map.')
if __name__ == '__main__':
    try:
        generate_processmap()
    except Exception as e:
         st.write("please try again")
         st.stop()