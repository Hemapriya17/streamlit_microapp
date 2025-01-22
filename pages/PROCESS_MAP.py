import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import requests
import io
import base64
import zlib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Initialize OpenAI client with API key
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def generate_mermaid_diagram(process_name):
    prompt = f"Generate a simple Mermaid flowchart diagram code for the process '{process_name}'. Use only basic flowchart syntax with rectangles and arrows. Do not include any explanations or markdown code blocks, just the raw Mermaid code."
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates simple Mermaid flowchart diagram code without any markdown formatting."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            n=1,
            temperature=0
        )
        mermaid_code = response.choices[0].message.content.strip()
        
        # Remove any markdown code block indicators and 'mermaid' tag if present
        mermaid_code = mermaid_code.replace('```mermaid', '').replace('```', '').strip()
        
        logging.info(f"Generated Mermaid code: {mermaid_code}")
        return mermaid_code
    except Exception as e:
        logging.error(f"An error occurred while generating the Mermaid diagram: {str(e)}")
        st.error(f"An error occurred while generating the Mermaid diagram: {str(e)}")
        return None

def get_mermaid_image(mermaid_code):
    try:
        # Compress the Mermaid code
        compressed = zlib.compress(mermaid_code.encode('utf-8'))
        
        # Base64 encode the compressed data
        encoded = base64.urlsafe_b64encode(compressed).decode('ascii')
        
        # Create the URL for the Kroki API
        url = f"https://kroki.io/mermaid/png/{encoded}"
        
        logging.info(f"Kroki URL: {url}")
        
        # Fetch the image
        response = requests.get(url)
        
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        else:
            logging.error(f"Failed to fetch the process map image. Status code: {response.status_code}")
            logging.error(f"Response content: {response.content}")
            st.error(f"Failed to fetch the process map image. Status code: {response.status_code}")
            st.error(f"Mermaid code causing the error: {mermaid_code}")
            return None
    except Exception as e:
        logging.error(f"An error occurred while fetching the image: {str(e)}")
        st.error(f"An error occurred while fetching the image: {str(e)}")
        return None

def generate_processmap():
    st.title("Process Map Generator")
    
    processmap_name = st.text_input(
        'Enter the process (e.g., Cleaning a coffee maker, making coffee using a coffee maker):',
        placeholder='Enter the process to generate a map'
    )

    if st.button("Generate Process Map"):
        if not processmap_name.strip():
            st.error("Please provide a valid input.")
            return
        
        with st.spinner("Generating process map..."):
            mermaid_code = generate_mermaid_diagram(processmap_name)
            
            if mermaid_code:
                # st.text("Mermaid code generated successfully. Fetching image...")
                image = get_mermaid_image(mermaid_code)
                if image:
                    # st.subheader("Generated Process Map:")
                    st.image(image, caption="Process Map", use_column_width=True)
                else:
                    st.error("Failed to generate the process map image. Please check the logs for more information.")
            else:
                st.error("Failed to generate the Mermaid diagram. Please check the logs for more information.")

# Call the function to generate the process map
if __name__ == "__main__":
    generate_processmap()
