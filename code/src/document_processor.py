import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st

def web_loader(doc_lis):
    loader = WebBaseLoader(
            web_paths=doc_lis,
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("hero__text-section-container col col-12 col-lg-6", "site-main")
                )
            ),
        )
    docs = loader.load()
    return docs
    
def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    return text_splitter.split_documents(docs)
        
@st.cache_data
def process_docs(docs_lis):
    docs = web_loader(docs_lis)
    docs = split_text(docs)
    return docs
