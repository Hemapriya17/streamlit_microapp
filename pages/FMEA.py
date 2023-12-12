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
import time
from zipfile import ZipFile
import base64

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])  # this is also the default, it can be omitted)

def generate_fmea():
    
        # print('vezbezb')
        fmea_name = st.text_input('NAME',' ',placeholder = 'Enter the name for which you would like to generate FMEA table',key = 'fmea_name')
        col1, col2 = st.columns([1,1])
        with col1:
            tmp_button = st.button(label='Submit')
        
        # st.session_state.result = ''
        print(tmp_button,fmea_name)
        if tmp_button and fmea_name:
            prompt = f'generate full design FMEA for {fmea_name} in a table format and add atleast 3 failure modes for each of the item/function?. Only return the generated FMEA table and don"t return any unwanted texts.'
            messages=[{"role": "user", "content": prompt}]
            start_time = time.time()
            completion = client.chat.completions.create(model="gpt-4-1106-preview",messages=messages,temperature = 0)
            end_time = time.time()

            time_lapsed = end_time - start_time

            st.write(f'{round(time_lapsed, 2)} secs')
            
            # print('bezbezb')
            response = completion.choices[0].message.content
            # st.write(response)
            # print(response)
            st.session_state.response = response
            # st.write(response)
            st.session_state.result = 'done'
        #     zipObj = ZipFile("sample.zip", "w")
        #     with open('myfile.txt', 'w') as fp: 
        #         pass

        #     # Add multiple files to the zip
        #     zipObj.write("myfile.txt")
        #     # zipObj.write("raportO.csv")
        #     # close the Zip File
        #     zipObj.close()

        #     ZipfileDotZip = "sample.zip"
        #     with open(ZipfileDotZip, "rb") as fp:
        #         btn = st.download_button(
        #     label="Download ZIP",
        #     data=fp,
        #     file_name="myfile.zip",
        #     mime="application/zip"
        # )
        if st.session_state.get('result','') and st.session_state.get('response',''):
            st.write(st.session_state.response)
        with col2:
            dowmload = st.download_button('Download',st.session_state.get('response',''),file_name = f'{fmea_name}_FMEA.txt')

def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

if __name__ == '__main__':
    try:
        generate_fmea()
    except Exception as e:
        st.stop()
    # export_as_pdf = st.button("Export Report")



    # if export_as_pdf:
    #     pdf = FPDF()
    #     pdf.add_page()
    #     pdf.set_font('Arial', 'B', 16)
    #     pdf.cell(40, 10, 'vewbvewb')
        
    #     html = create_download_link(pdf.output(dest="S").encode("latin-1"), "test")

    #     st.markdown(html, unsafe_allow_html=True)
