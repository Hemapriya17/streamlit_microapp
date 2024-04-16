# import os
# from openai import OpenAI
# import streamlit as st
# from dotenv import load_dotenv
# load_dotenv()
# import json
# import pandas as pd
# import ast
# import time
# import openai
# import time
# from zipfile import ZipFile
# import base64
# import requests

# client = OpenAI(api_key='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')  # this is also the default, it can be omitted)
# api_key = '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'
# def encode_image(image_file):
#     image_data = image_file.getvalue()
#     return base64.b64encode(image_data).decode('utf-8')

# def clear_cache():
#     st.cache_data.clear()
#     st.session_state.fmea_name = ''
#     # st.session_state.response_df =''
#     # variable_values.clear()
#     # # st.cache_resource.clear()
#     # st.session_state.analyseDoe = ''
#     # st.session_state.click = False
# def generate_fmea():
#         st.title("OpenAI Image ")

#     # File uploader
#         image_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg'])
#         response = ''
#         if image_file is not None:
#             # Display the image
#             # st.image(image_file, caption='Uploaded Image', use_column_width=True)

#             # Getting the base64 string
#             base64_image = encode_image(image_file)

#             headers = {
#                 "Content-Type": "application/json",
#                 "Authorization": f"Bearer {api_key}"
#             }

#             payload = {
#                 "model": "gpt-4-vision-preview",
#                 "messages": [
#                 {
#                     "role": "user",
#                     "content": [
#                     {
#                         "type": "text",
#                         "text": "Describe the product/appliance present in the image and also return the what specific product is in the image. Answer shoud be in tuple where first element is description and the other element is the name of the main componenet. For the main component return only specific component name and don't return any other text."
#                     },
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                         "url": f"data:image/jpeg;base64,{base64_image}"
#                         }
#                     }
#                     ]
#                 }
#                 ],
#                 "max_tokens": 300
#             }

#             response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
#         # print('vezbezb')
#         if response:
#             response = response.json()
#             fmea_name = response.get('choices',[])[0].get('message',{}).get('content')
#             print(fmea_name)
#             fmea_name = ast.literal_eval(str(fmea_name))

#             print(type(fmea_name))
            
#             st.session_state.fmea_name = fmea_name[1]
#             st.write(fmea_name)
#             # fmea_product = fmea_name[1]
#         fmea_name = st.text_input('NAME','',placeholder = 'Enter the name for which you would like to generate FMEA table',key = st.session_state.fmea_name)
#         # print(fmea_name)
#         col1, col2 = st.columns([1,1])
#         with col1:
#             tmp_button = st.button(label='Submit')
#         # enc = encode_image('pages/toaster.jpg')
#         # print(enc)
#         st.session_state.result = ''
#         st.session_state.fmea_name = fmea_name
#         # print(tmp_button,fmea_name)
#         if tmp_button and st.session_state.fmea_name:
#             # prompt = f'What product is in the pages/water_heater1.jpg.Response should be like ebike, mobile phone etc. and the answer should be in a single word'
#             # messages=[{"role": "user", "content": prompt}]
#             # completion = client.chat.completions.create(model="gpt-4-vision-preview",messages=messages,temperature = 0)
#             # image_response = completion.choices[0].message.content
#             # # start_time = time.time()
#             # print(image_response)
#             st.write(fmea_name)
#             prompt = f'generate full design FMEA for {fmea_name} in a table format and add atleast 3 failure modes for each of the item/function?. Only return the generated FMEA table and don"t return any unwanted texts.'
#             messages=[{"role": "user", "content": prompt}]
#             start_time = time.time()
#             completion = client.chat.completions.create(model="gpt-4-1106-preview",messages=messages,temperature = 0)
#             end_time = time.time()

#             time_lapsed = end_time - start_time

#             st.write(f'{round(time_lapsed, 2)} secs')
            
#             # print('bezbezb')
#             response = completion.choices[0].message.content
#             # st.write(response)
#             # print(response)
#             st.session_state.response = response
#             # st.write(response)
#             st.session_state.result = 'done'
#         #     zipObj = ZipFile("sample.zip", "w")
#         #     with open('myfile.txt', 'w') as fp: 
#         #         pass

#         #     # Add multiple files to the zip
#         #     zipObj.write("myfile.txt")
#         #     # zipObj.write("raportO.csv")
#         #     # close the Zip File
#         #     zipObj.close()

#         #     ZipfileDotZip = "sample.zip"
#         #     with open(ZipfileDotZip, "rb") as fp:
#         #         btn = st.download_button(
#         #     label="Download ZIP",
#         #     data=fp,
#         #     file_name="myfile.zip",
#         #     mime="application/zip"
#         # )
#         if st.session_state.get('result','') and st.session_state.get('response',''):
#             st.write(st.session_state.response)
#         with col2:
#             # dowmload = st.download_button('Download',st.session_state.get('response',''),file_name = f'{fmea_name}_FMEA.txt')
#             clear_button = st.button("Clear",on_click=clear_cache)

# def create_download_link(val, filename):
#     b64 = base64.b64encode(val)  # val looks like b'...'
#     return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

# if __name__ == '__main__':
#     # try:
#         generate_fmea()
#     # except Exception as e:
#     #     st.stop()
#     # export_as_pdf = st.button("Export Report")



# # # #     # if export_as_pdf:
# # # #     #     pdf = FPDF()
# # # #     #     pdf.add_page()
# # # #     #     pdf.set_font('Arial', 'B', 16)
# # # #     #     pdf.cell(40, 10, 'vewbvewb')
        
# # # #     #     html = create_download_link(pdf.output(dest="S").encode("latin-1"), "test")

# # # #     #     st.markdown(html, unsafe_allow_html=True)

# # # # """Example usage of GPT4-V API.

# # # # Usage:

# # # #     OPENAI_API_KEY=<your_api_key> python3 gpt4v.py \
# # # #         [<path/to/image1.png>] [<path/to/image2.jpg>] [...] "text prompt"

# # # # Example:

# # # #     OPENAI_API_KEY=xxx python3 gpt4v.py photo.png "What's in this photo?"
# # # # """

# # # # from pprint import pprint
# # # # import base64
# # # # import json
# # # # import mimetypes
# # # # import os
# # # # import requests
# # # # import sys


# # # # api_key = os.getenv("OPENAI_API_KEY")


# # # # def encode_image(image_file):
# #     # image_data = image_file.getvalue()
# #     # return base64.b64encode(image_data).decode('utf-8')


# # # # def create_payload(images: list[str], prompt: str, model="gpt-4-vision-preview", max_tokens=100, detail="high"):
# # # #     """Creates the payload for the API request."""
# # # #     messages = [
# # # #         {
# # # #             "role": "user",
# # # #             "content": [
# # # #                 {
# # # #                     "type": "text",
# # # #                     "text": prompt,
# # # #                 },
# # # #             ],
# # # #         },
# # # #     ]
# # # #     if images:
# # # #         for image in images:
# # # #             base64_image = encode_image(image)
# # # #             messages[0]["content"].append({
# # # #                 "type": "image_url",
# # # #                 "image_url": {
# # # #                     "url": base64_image,
# # # #                     "detail": detail,
# # # #                 }
# # # #             })

# # # #     return {
# # # #         "model": model,
# # # #         "messages": messages,
# # # #         "max_tokens": max_tokens
# # # #     }


# # # # def query_openai(payload):
# # # #     """Sends a request to the OpenAI API and prints the response."""
# # # #     headers = {
# # # #         "Content-Type": "application/json",
# # # #         "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
# # # #     }
# # # #     response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
# # # #     return response.json()


# # # # def main():
# # # #     # if len(sys.argv) < 3:
# # # #     #     print("Usage: python script.py [image1.jpg] [image2.png] ... \"Text Prompt\"")
# # # #     #     sys.exit(1)

# # # #     # All arguments except the last one are image paths
# # # #     print(sys.argv)
# # # #     if len(sys.argv) >2:

# # # #         image_paths = sys.argv[1:-1]
# # # #     else:
# # # #         image_paths = ''
# # # #     # The last argument is the text prompt
# # # #     prompt = sys.argv[-1]

# # # #     payload = create_payload(image_paths, prompt)
# # # #     response = query_openai(payload)
# # # #     pprint(response)


# # # # if __name__ == "__main__":
# # # #     main()


# # import base64
# # import requests
# # import streamlit as st
# # from io import BytesIO

# # # OpenAI API Key
# # api_key = "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"

# # # Function to encode the image
# # def encode_image(image_file):
# #     image_data = image_file.getvalue()
# #     return base64.b64encode(image_data).decode('utf-8')

# # def main():
# #     st.title("OpenAI Image Chat")

# #     # File uploader
# #     image_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg'])

# #     if image_file is not None:
# #         # Display the image
# #         st.image(image_file, caption='Uploaded Image', use_column_width=True)

# #         # Getting the base64 string
# #         base64_image = encode_image(image_file)

# #         headers = {
# #             "Content-Type": "application/json",
# #             "Authorization": f"Bearer {api_key}"
# #         }

# #         payload = {
# #             "model": "gpt-4-vision-preview",
# #             "messages": [
# #               {
# #                 "role": "user",
# #                 "content": [
# #                   {
# #                     "type": "text",
# #                     "text": "Describe image in single word"
# #                   },
# #                   {
# #                     "type": "image_url",
# #                     "image_url": {
# #                       "url": f"data:image/jpeg;base64,{base64_image}"
# #                     }
# #                   }
# #                 ]
# #               }
# #             ],
# #             "max_tokens": 300
# #         }

# #         response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

# #         if response.status_code == 200:
# #             st.success("Description generated successfully:")
# #             st.write(response.json())
# #         else:
# #             st.error("Failed to generate description, please try again.")

# # if __name__ == "__main__":
# #     main()


import os
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv
import json
import pandas as pd
import ast
import time
import openai
import time
from zipfile import ZipFile
import base64
import requests

# Load environment variables
load_dotenv()

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = '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'

# Initialize OpenAI client
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Function to encode image
def encode_image(image_file):
    image_data = image_file.getvalue()
    return base64.b64encode(image_data).decode('utf-8')

# Function to clear cache
def clear_cache():
    st.cache_data.clear()
    st.session_state.fmea_name = ''
    st.session_state.image_response_cache = {}

# Function to generate FMEA
def generate_fmea():
    if 'fmea_name' not in st.session_state:
        st.session_state.fmea_name = ''
    if 'image_response_cache' not in st.session_state:
        st.session_state.image_response_cache = {}
    
    st.title("FMEA")

    # File uploader
    image_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg'])
    response = ''

    if image_file is not None:
        # Encode the image
        base64_image = encode_image(image_file)

        # Check if the image response is cached
        if base64_image in st.session_state.image_response_cache:
            response = st.session_state.image_response_cache[base64_image]
        else:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
            }

            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe the product/appliance present in the image and also return the what specific product is menioned in the description before. Answer shoud be in tuple where first element is description and the other element is the name of the main componenet. For the main component return only specific component name and don't return any other text."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 300
            }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            if response.status_code == 200:
                response = response.json()
                st.session_state.image_response_cache[base64_image] = response.get('choices',[])[0].get('message',{}).get('content')
    
    if response:
        fmea_name = st.session_state.image_response_cache.get(base64_image)
        fmea_name = ast.literal_eval(str(fmea_name))
        st.session_state.fmea_name = fmea_name[0]
        # st.write(fmea_name)
    default = st.session_state.get('fmea_name','')
    if isinstance(default,tuple):
        default = default[0]
    else:
        default = ''
    fmea_name = st.text_input('FMEA Description',st.session_state.fmea_name,placeholder='Enter the name for which you would like to generate FMEA table')
    
    col1, col2 = st.columns([1,1])

    with col1:
        tmp_button = st.button(label='Submit')

    st.session_state.result = ''
    # if not isinstance(st.session_state.fmea_name,list):

    st.session_state.fmea_name = fmea_name

    if tmp_button and st.session_state.fmea_name:
        prompt = f'generate full design FMEA for {st.session_state.fmea_name} in a table format and add atleast 3 failure modes for each of the item/function?. Only return the generated FMEA table and don"t return any unwanted texts.'
        # st.write(prompt)
        messages = [{"role": "user", "content": prompt}]
        start_time = time.time()
        # completion = client.chat.completions.create(model="gpt-4-1106-preview",messages=messages,temperature=0)
        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
    }

        completion = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json={"model": "gpt-4-1106-preview", "messages": messages, "temperature": 0})
        end_time = time.time()

        time_lapsed = end_time - start_time

        st.write(f'{round(time_lapsed, 2)} secs')
        
        # response = completion.choices[0].message.content
        completion = completion.json()
        response = completion.get('choices', [])[0].get('message', {}).get('content')
        
        st.session_state.response = response
        st.session_state.result = 'done'

    if st.session_state.get('result','') and st.session_state.get('response',''):
        st.write(st.session_state.response)

    with col2:
        clear_button = st.button("Clear",on_click=clear_cache)

if __name__ == '__main__':
    try:
        generate_fmea()
    except Exception as e:
        st.stop()
