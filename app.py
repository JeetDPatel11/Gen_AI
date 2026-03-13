import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -------------------------------
# Load Secrets (Streamlit Cloud)
# -------------------------------

os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
os.environ["LANGSMITH_PROJECT"] = st.secrets["LANGSMITH_PROJECT"]
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# -------------------------------
# Create LLM
# -------------------------------

llm = ChatGroq(
    model="llama3-8b-8192"
)

# -------------------------------
# Prompt Template
# -------------------------------

prompt = ChatPromptTemplate.from_template(
    "Answer the following question: {question}"
)

# -------------------------------
# LangChain
# -------------------------------

chain = LLMChain(
    llm=llm,
    prompt=prompt
)

# -------------------------------
# Streamlit UI
# -------------------------------

st.title("Gen AI Q&A App")

input_text = st.text_input("What question do you have in mind?")

if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response["text"])