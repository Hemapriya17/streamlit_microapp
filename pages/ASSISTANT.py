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


# client = openai.OpenAI()
client = OpenAI(api_key='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')  # this is also the default, it can be omitted)

model = "gpt-4-1106-preview"  # "gpt-3.5-turbo-16k"

# == Hardcoded ids to be used once the first code run is done and the assistant was created
thread_id = "thread_jNEm1hTkmk1gRqJgiM8vnUEn" # thread_HrPFFFNyatwPvTXEMkFlUBiu       thread_FRETt7XW74AGkuWWUlGGktns
assis_id = "asst_WCFObajzpLIdQjmT2LZZinEu"  # asst_WCFObajzpLIdQjmT2LZZinEu           asst_qVxgM3IhddbqSXAoLRlVyKEv


if "start_chat" not in st.session_state:
    st.session_state.start_chat = False

if "thread_id" not in st.session_state:
    st.session_state.thread_id = thread_id




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
                    print('Generated Mermaid Code:', graph)  # Debugging line
                    if graph.strip():  # Check if graph is not empty
                        components.html(
                            f"""
                            <pre class="mermaid">
                                {graph}
                            </pre>

                            <script type="module">
                                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                                mermaid.initialize({{ startOnLoad: true }});
                            </script>
                            """,
                            width=1000,
                            height=150,
                            scrolling=True
                        )
                    else:
                        st.error("No valid Mermaid code generated.")
   
    else:
        # Prompt users to start chat
        # st.write("Please start the chat by clicking the button on the sidebar.")
        pass

