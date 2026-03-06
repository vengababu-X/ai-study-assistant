import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

st.title("AI Study Assistant")

documents = []

for root, dirs, files in os.walk("pdfs"):
    for file in files:
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(root, file))
            documents.extend(loader.load())

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
