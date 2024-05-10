__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain_openai import ChatOpenAI
from document_pages import PAGES
from document_retriever import build_retriever
from document_retriever_simple import build_simple_retriever
from rag import ConversationalRAGChain,ChatHistoryAwareRetriever, QAChain
from document_processor import process_docs
import dotenv
import os

dotenv.load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
docs = process_docs(PAGES)
dretriever = build_simple_retriever(docs)
history_aware_retriever = ChatHistoryAwareRetriever(llm, dretriever)
qa_chain = QAChain(llm)
qa_rag = ConversationalRAGChain(history_aware_retriever.retriever, qa_chain.question_answer_chain)
st.session_state.qa_rag = qa_rag

st.title("💬 SaaS Management QA BOT")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    response = st.session_state.qa_rag.get_answer(prompt)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
