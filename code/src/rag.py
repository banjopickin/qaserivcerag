from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


class ChatHistoryAwareRetriever(object):
    def __init__(self, llm, document_retreiver) -> None:
        contextualized_q_prompt = self.build_prompt()
        self.retriever = create_history_aware_retriever(llm, document_retreiver, contextualized_q_prompt)

    def system_prompt(self,):
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
        return contextualize_q_system_prompt

    def build_prompt(self,):
        system_prompt = self.system_prompt()
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt),
             MessagesPlaceholder("chat_history"),
             ("human", "{input}"),
            ])
        return contextualize_q_prompt
    
class QAChain(object):
    def __init__(self,llm) -> None:
        qa_prompt = self.build_prompt()
        self.question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    def system_prompt(self,):
        return '''
        Only Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that I am sorry, I don't know, don't try to make up an answer.
        Use monden English and business tone. Keep the answer as concise as possible.

        {context}
        '''
    
    def build_prompt(self):
        system_prompt = self.system_prompt()
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        return qa_prompt
    
class ConversationalRAGChain(object):
    def __init__(self,history_aware_retriever,question_answer_chain) -> None:
        self.history_aware_retriever = history_aware_retriever
        self.question_answer_chain = question_answer_chain
        self.session_store = {}
        self.chain = self.build_conversational_rag_chain()

    def build_rag_chain(self):
        return create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain)
    
    def get_session_history(self,session_id: str, latest_k = 6) -> BaseChatMessageHistory:
        if session_id not in self.session_store:
            self.session_store[session_id] = ChatMessageHistory()
            return self.session_store[session_id]
        else:
            history = self.session_store[session_id]
            return ChatMessageHistory(messages = history.messages[-latest_k:])
    
    def build_conversational_rag_chain(self,):
        rag_chain = self.build_rag_chain()
        return RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def get_answer(self, question):
        response = self.chain.invoke(
            {"input": question},
            config={
                "configurable": {"session_id": "abc123"}
            })
        return response["answer"]
