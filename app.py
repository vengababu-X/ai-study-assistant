import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

st.title("AI Study Assistant")

documents = []

# Load all PDFs in repository
for file in os.listdir():
    if file.endswith(".pdf"):
        loader = PyPDFLoader(file)
        documents.extend(loader.load())

# Create vector database
db = Chroma.from_documents(documents, OpenAIEmbeddings())

retriever = db.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=retriever
)

question = st.text_input("Ask a question from your notes")

if question:
    answer = qa.run(question)
    st.write(answer)
