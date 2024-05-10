from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import streamlit as st

@st.cache_resource
def build_simple_retriever(_docs):
    vectorstore = Chroma.from_documents(documents=_docs, embedding=OpenAIEmbeddings(model = "text-embedding-ada-002"))
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    return retriever
