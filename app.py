import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------
# Load Secrets (Streamlit Cloud)
# ---------------------------------

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
os.environ["LANGSMITH_PROJECT"] = st.secrets["LANGSMITH_PROJECT"]
os.environ["LANGSMITH_TRACING"] = "true"

# ---------------------------------
# Initialize LLM
# ---------------------------------

llm = ChatGroq(
    model="llama-3.1-8b-instant"
)

# ---------------------------------
# Prompt Template
# ---------------------------------

prompt = ChatPromptTemplate.from_template(
    "Answer the following question clearly: {question}"
)

# ---------------------------------
# LangChain Pipeline
# ---------------------------------

chain = prompt | llm | StrOutputParser()

# ---------------------------------
# Streamlit UI
# ---------------------------------

st.title("Gen AI Q&A App")

st.write("Ask any question and the AI will respond.")

input_text = st.text_input("What question do you have in mind?")

if input_text:
    with st.spinner("Generating response..."):
        response = chain.invoke({"question": input_text})
        st.write(response)