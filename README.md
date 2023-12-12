## Installation

Follow the instructions below to run the Streamlit server locally.

### Pre-requisites

Make sure you have Python ≥3.10 installed.

### Steps

1. Clone the repository

```bash
git clone https://github.com/Testrunz/streamlit_microapp.git
cd streamlit_microapp
```

2. Install dependencies from requirements file 

```bash
pip install -r requirements.txt
```

3. (Optional) Avoid adding the OpenAI API every time you run the server by adding it to environment variables.
   - Make a copy of `.env.example` named `.env`
   - Add your API key to the `.env` file

> **Note:** Make sure you have a paid OpenAI API key for faster completions and to avoid hitting rate limits.

4. Run the Streamlit server

```bash
streamlit run main_app.py
```