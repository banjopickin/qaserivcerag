from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.storage import InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
import streamlit as st


def summary_chain():
    chain = (
        {"doc": lambda x: x.page_content}
        | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
        | ChatOpenAI(max_retries=0)
        | StrOutputParser()
    )
    return chain

def multi_vector_retriever(k=6):
    vectorstore = Chroma(collection_name="document", embedding_function=OpenAIEmbeddings(model="text-embedding-ada-002"))
    # The storage layer for the parent documents
    store = InMemoryByteStore()
    # The retriever (empty to start)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key="doc_id",
        search_kwargs = {"k":k}
        )
    return retriever

@st.cache_resource
def build_retriever(_docs):
    doc_ids = [str(uuid.uuid4()) for _ in _docs]
    chain = summary_chain()
    summaries = chain.batch(_docs, {"max_concurrency":5})
    summary_docs = [Document(page_content=s, metadata={"doc_id": doc_ids[i]}) for i, s in enumerate(summaries)]
    retriever = multi_vector_retriever()
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, _docs)))
    return retriever
