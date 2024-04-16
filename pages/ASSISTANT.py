import os
import openai
import requests
import json
from openai import OpenAI
import time
import logging
from datetime import datetime
import streamlit as st
import base64
import requests
import io
from PIL import Image
import streamlit.components.v1 as components

# Initialize the OpenAI client with your API key
openai.api_key = "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"

# Load environment variables from .env file
# load_dotenv()

# client = openai.OpenAI()
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])  # this is also the default, it can be omitted)

model = "gpt-4-1106-preview"  # "gpt-3.5-turbo-16k"

# == Hardcoded ids to be used once the first code run is done and the assistant was created
thread_id = "thread_jNEm1hTkmk1gRqJgiM8vnUEn" # thread_HrPFFFNyatwPvTXEMkFlUBiu       thread_FRETt7XW74AGkuWWUlGGktns
assis_id = "asst_WCFObajzpLIdQjmT2LZZinEu"  # asst_WCFObajzpLIdQjmT2LZZinEu           asst_qVxgM3IhddbqSXAoLRlVyKEv

# Initialize all the session
# if "file_id_list" not in st.session_state:
#     st.session_state.file_id_list = []

if "start_chat" not in st.session_state:
    st.session_state.start_chat = False

if "thread_id" not in st.session_state:
    st.session_state.thread_id = thread_id


# # Set up our front end page
# st.set_page_config(page_title="Microapps")


# # ==== Function definitions etc =====
# def upload_to_openai(filepath):
#     with open(filepath, "rb") as file:
#         response = client.files.create(file=file.read(), purpose="assistants")
#     return response.id


# # === Sidebar - where users can upload files
# file_uploaded = st.sidebar.file_uploader(
#     "Upload a file to be transformed into embeddings", key="file_upload"
# )

# # Upload file button - store the file ID
# if st.sidebar.button("Upload File"):
#     if file_uploaded:
#         with open(f"{file_uploaded.name}", "wb") as f:
#             f.write(file_uploaded.getbuffer())
#         another_file_id = upload_to_openai(f"{file_uploaded.name}")
#         st.session_state.file_id_list.append(another_file_id)
#         st.sidebar.write(f"File ID:: {another_file_id}")

# # Display those file ids
# if st.session_state.file_id_list:
#     st.sidebar.write("Uploaded File IDs:")
#     for file_id in st.session_state.file_id_list:
#         st.sidebar.write(file_id)

# Button to initiate the chat session
if st.sidebar.button("Start Chatting..."):
    # if st.session_state.file_id_list:
    st.session_state.start_chat = True

    # Create a new thread for this chat session
    chat_thread = client.beta.threads.create()
    st.session_state.thread_id = st.session_state.thread_id
    st.write("Thread ID:", st.session_state.thread_id)
    # else:
    #     st.sidebar.warning(
    #         "No files found. Please upload at least one file to get started."
    #     )


# Define the function to process messages with citations
def process_message_with_citations(message):
    """Extract content and annotations from the message and format citations as footnotes."""
    message_content = message.content[0].text
    annotations = (
        message_content.annotations if hasattr(message_content, "annotations") else []
    )
    citations = []

    # Iterate over the annotations and add footnotes
    for index, annotation in enumerate(annotations):
        # Replace the text with a footnote
        message_content.value = message_content.value.replace(
            annotation.text, f" [{index + 1}]"
        )

        # Gather citations based on annotation attributes
        if file_citation := getattr(annotation, "file_citation", None):
            # Retrieve the cited file details (dummy response here since we can't call OpenAI)
            cited_file = {
                "filename": "cryptocurrency.pdf"
            }  # This should be replaced with actual file retrieval
            citations.append(
                f'[{index + 1}] {file_citation.quote} from {cited_file["filename"]}'
            )
        elif file_path := getattr(annotation, "file_path", None):
            # Placeholder for file download citation
            cited_file = {
                "filename": "cryptocurrency.pdf"
            }  # TODO: This should be replaced with actual file retrieval
            citations.append(
                f'[{index + 1}] Click [here](#) to download {cited_file["filename"]}'
            )  # The download link should be replaced with the actual download path

    # Add footnotes to the end of the message content
    full_response = message_content.value + "\n\n" + "\n".join(citations)
    return full_response


# the main interface ...
# st.title("Study Buddy")
# st.write("Learn fast by chatting with your documents")


# Check sessions
if st.session_state.start_chat:
    if "openai_model" not in st.session_state:
        st.session_state.openai_model = "gpt-4-1106-preview"
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show existing messages if any...
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # chat input for the user
    if prompt := st.chat_input("What's new?"):
        # Add user message to the state and display on the screen
        # prompt = f'Generate mindmap  (mermaid code) for {prompt} process using GoT principles. Don"t return any numbers or any unwanted text, description of the response. Don"t return flowchart diagram.'
        # prompt = f'{prompt} using GoT(Graph of Thought) principles. Return the mermaid code and don"t return any other texts and don"t the mermaid code as flowchart. Start the mermaid code response with # symbol and end the mermaid code with # symbol'
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # add the user's message to the existing thread
        client.beta.threads.messages.create(
            thread_id=st.session_state.thread_id, role="user", content=prompt
        )

        # Create a run with additional instructions
        run = client.beta.threads.runs.create(
            thread_id=st.session_state.thread_id,
            assistant_id=assis_id,
            instructions="""You are a helpful assistant who creates mermaid code using the Graph of Thought concept and  using the below instruction.
Identify the main goal from the problem statement.
Break down the main goal into smaller, manageable tasks or sub-goals. Atleast comeup with 5 sub goals. This is a key step. Do not proceed sequencially. 
For each task or sub-goal, identify any dependencies or prerequisites needed before it can be started.
If multiple methods exist for achieving a sub-goal, outline these as parallel paths. Atleast comeup with 5 parallel paths.
Show dependencies of these parallel paths if answer to one of the path is an input to the other.
Assume highly probable answers to some of these parallel paths to help converge. 
Determine how these paths will converge as we approach the final goal.
Review the entire process for any potential refinements or optimizations.
Outline the final steps towards achieving the end goal.
Only Generate a mermaid code to show this graph of thought and no texts.
Merge all the paths and give me one or two output nodes

Only generate mermaid code to show the graph of thought and no texts.


Remember, a Graph of Thoughts is a dynamic tool, so it should be adaptable as new information is introduced or as goals evolve.""",
        )

        # Show a spinner while the assistant is thinking...
        with st.spinner("Wait... Generating response..."):
            while run.status != "completed":
                time.sleep(1)
                run = client.beta.threads.runs.retrieve(
                    thread_id=st.session_state.thread_id, run_id=run.id
                )
            # Retrieve messages added by the assistant
            messages = client.beta.threads.messages.list(
                thread_id=st.session_state.thread_id
            )
            # Process and display assistant messages
            assistant_messages_for_run = [
                message
                for message in messages
                if message.run_id == run.id and message.role == "assistant"
            ]

            for message in assistant_messages_for_run:
                full_response = process_message_with_citations(message=message)
                print('full_response',full_response)

                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
                with st.chat_message("assistant"):
                    response = full_response.split('\n',1)[1]
                    print('dict_response',response)
                    graph = response.rsplit('```',1)[0]
                    print('graph',graph)
                    components.html(
        f"""
        <pre class="mermaid">
            {graph}
        </pre>

        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        """,width=1000,
        height = 150,
        scrolling=True
    )
                    # st.write(response)
                    # graphbytes = graph.encode("ascii")
                    # base64_bytes = base64.b64encode(graphbytes)
                    # base64_string = base64_bytes.decode("ascii")

                    # print(base64_string)
                    # print('request',requests.get('https://mermaid.ink/img/' + base64_string))
                    # img = Image.open(io.BytesIO(requests.get('https://mermaid.ink/img/' + base64_string).content))
                    # response = full_response.split("```",1)[1]
                    # print('dict_response',response)
                    # graph = response.rsplit("```",1)[0]
                    # response = graph.split("mermaid",1)[1]
                    # print('response',response)
                    # graphbytes = response.encode("ascii")
                    # base64_bytes = base64.b64encode(graphbytes)
                    # base64_string = base64_bytes.decode("ascii")

                    # # print(base64_string)
                    # print('base64_string',base64_string)
                    # img = Image.open(io.BytesIO(requests.get('https://mermaid.ink/img/' + base64_string).content))
                    # image = Image.open(img)
                    # new_image = img.resize((700, 1000))
                    # st.image(new_image)
                    # st.image(img, caption='Processed Image')
                    # st.markdown(full_response, unsafe_allow_html=True)
    else:
        # Prompt users to start chat
        # st.write("Please start the chat by clicking the button on the sidebar.")
        pass



# import openai
# import streamlit as st
# import time

# # Initialize the OpenAI client with your API key
# openai.api_key = "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"

# # model = "gpt-4-1106-preview"  # "gpt-3.5-turbo-16k"

# # Hardcoded ids to be used once the first code run is done and the assistant was created
# thread_id = "thread_Z85N6eP0P5Plwi4YHqQrb2LK" #asst_qVxgM3IhddbqSXAoLRlVyKEv
# assis_id = "asst_qVxgM3IhddbqSXAoLRlVyKEv" #thread_Z85N6eP0P5Plwi4YHqQrb2LK

# # Initialize all the session
# if "file_id_list" not in st.session_state:
#     st.session_state.file_id_list = []

# if "start_chat" not in st.session_state:
#     st.session_state.start_chat = False

# if "thread_id" not in st.session_state:
#     st.session_state.thread_id = None

# # Set up our front end page
# st.set_page_config(page_title="Study Buddy - Chat and Learn", page_icon=":books:")

# # Function to upload file to OpenAI
# def upload_to_openai(filepath):
#     with open(filepath, "rb") as file:
#         response = openai.File.create(file=file.read(), purpose="assistants")
#     return response.id

# # === Sidebar - where users can upload files ===
# file_uploaded = st.sidebar.file_uploader(
#     "Upload a file to be transformed into embeddings", key="file_upload"
# )

# # Upload file button - store the file ID
# if st.sidebar.button("Upload File"):
#     if file_uploaded:
#         with open(f"{file_uploaded.name}", "wb") as f:
#             f.write(file_uploaded.getbuffer())
#         another_file_id = upload_to_openai(f"{file_uploaded.name}")
#         st.session_state.file_id_list.append(another_file_id)
#         st.sidebar.write(f"File ID:: {another_file_id}")

# # Display those file ids
# if st.session_state.file_id_list:
#     st.sidebar.write("Uploaded File IDs:")
#     for file_id in st.session_state.file_id_list:
#         st.sidebar.write(file_id)

# # Button to initiate the chat session
# if st.sidebar.button("Start Chatting..."):
#     if st.session_state.file_id_list:
#         st.session_state.start_chat = True

#         # Create a new thread for this chat session
#         chat_thread = openai.Thread.create()
#         st.session_state.thread_id = chat_thread.id
#         st.write("Thread ID:", chat_thread.id)
#     else:
#         st.sidebar.warning(
#             "No files found. Please upload at least one file to get started."
#         )

# # Define the function to process messages with citations
# def process_message_with_citations(message):
#     """Extract content and annotations from the message and format citations as footnotes."""
#     message_content = message.content[0].text
#     annotations = (
#         message_content.annotations if hasattr(message_content, "annotations") else []
#     )
#     citations = []

#     # Iterate over the annotations and add footnotes
#     for index, annotation in enumerate(annotations):
#         # Replace the text with a footnote
#         message_content.value = message_content.value.replace(
#             annotation.text, f" [{index + 1}]"
#         )

#         # Gather citations based on annotation attributes
#         if file_citation := getattr(annotation, "file_citation", None):
#             # Retrieve the cited file details (dummy response here since we can't call OpenAI)
#             cited_file = {
#                 "filename": "cryptocurrency.pdf"
#             }  # This should be replaced with actual file retrieval
#             citations.append(
#                 f'[{index + 1}] {file_citation.quote} from {cited_file["filename"]}'
#             )
#         elif file_path := getattr(annotation, "file_path", None):
#             # Placeholder for file download citation
#             cited_file = {
#                 "filename": "cryptocurrency.pdf"
#             }  # TODO: This should be replaced with actual file retrieval
#             citations.append(
#                 f'[{index + 1}] Click [here](#) to download {cited_file["filename"]}'
#             )  # The download link should be replaced with the actual download path

#     # Add footnotes to the end of the message content
#     full_response = message_content.value + "\n\n" + "\n".join(citations)
#     return full_response

# # Main interface
# st.title("Study Buddy")
# st.write("Learn fast by chatting with your documents")

# # Check sessions
# if st.session_state.start_chat:
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     # Show existing messages if any...
#     for message in st.session_state.messages:
#         with st.echo():
#             st.markdown(f"**{message['role']}**: {message['content']}")

#     # chat input for the user
#     if prompt := st.text_input("What's new?"):
#         # Add user message to the state and display on the screen
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.echo():
#             st.markdown(f"**User**: {prompt}")

#         # add the user's message to the existing thread
#         openai.Message.create(
#             thread_id=st.session_state.thread_id, role="user", content=prompt
#         )

#         # Create a run with additional instructions
#         run = openai.Run.create(
#             thread_id=st.session_state.thread_id,
#             assistant_id=assis_id,
#             instructions="""Please answer the questions using the knowledge provided in the files.
#             when adding additional information, make sure to distinguish it with bold or underlined text.""",
#         )

#         # Show a spinner while the assistant is thinking...
#         with st.spinner("Wait... Generating response..."):
#             while run.status != "completed":
#                 time.sleep(1)
#                 run = openai.Run.retrieve(
#                     thread_id=st.session_state.thread_id, run_id=run.id
#                 )
#             # Retrieve messages added by the assistant
#             messages = openai.Message.list(
#                 thread_id=st.session_state.thread_id
#             )
#             # Process and display assistant messages
#             assistant_messages_for_run = [
#                 message
#                 for message in messages
#                 if message.run_id == run.id and message.role == "assistant"
#             ]

#             for message in assistant_messages_for_run:
#                 full_response = process_message_with_citations(message=message)
#                 st.session_state.messages.append(
#                     {"role": "assistant", "content": full_response}
#                 )
#                 with st.echo():
#                     st.markdown(f"**Assistant**: {full_response}")
#     else:
#         # Prompt users to start chat
#         st.write("Please start the chat by clicking the button on the sidebar.")
