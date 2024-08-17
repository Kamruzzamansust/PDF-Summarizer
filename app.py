import os
from dotenv import load_dotenv
import streamlit as st
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Set up the LLM with GROQ API
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it", streaming=True)

# Define prompts
chunks_prompt = """
Please summarize the below text:
`{text}'
Summary:
"""
map_prompt_template = PromptTemplate(input_variables=['text'], template=chunks_prompt)

final_prompt = '''
Provide the final summary of the entire book with these important points.
Add a information Title, Start the precise summary with an introduction and provide the summary in number 
points for the books.
docs:{text}
'''
final_prompt_template = PromptTemplate(input_variables=['text'], template=final_prompt)

# Initialize the summarize chain
summary_chain = load_summarize_chain(
    llm=llm,
    chain_type="map_reduce",
    map_prompt=map_prompt_template,
    combine_prompt=final_prompt_template,
    verbose=True
)

# Streamlit App
st.title("PDF Summarizer")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

import tempfile

if uploaded_file is not None:
    with st.spinner("Processing the PDF..."):
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        
        # Load and split the PDF document
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load_and_split()
        
        final_documents = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=100
        ).split_documents(docs)
        
        # Run the summarization chain
        summary = summary_chain.run(final_documents)
        
        st.success("Summary generated successfully!")
        st.write(summary)
        # The rest of your code remains unchanged
