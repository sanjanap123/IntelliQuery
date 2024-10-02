from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq  # Import the model you're going to use
from langchain.schema import HumanMessage 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.globals import set_debug
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma  # Correct import for Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st
import chromadb
import getpass
import chromadb

import os
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)

def set_api(api_key:str):
        os.environ["Groq_API_KEY"]=api_key
 
def create_vector_store_retriver(file_paths:list[str],chunk_size:int=1000, chunk_overlap:int=200):
        all_splits = []
        doc_list = []
        for file_path in file_paths:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            doc_list.extend(docs)

            #splits = text_splitter.split_documents(docs)
            #all_splits.extend(splits)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
        all_splits = text_splitter.split_documents(doc_list)

        vectorstore = Chroma.from_documents(documents=all_splits, embedding=HuggingFaceEmbeddings())
        retriever = vectorstore.as_retriever()
        return retriever
    
# def vector_retr(documents):
#     vectorstore = Chroma.from_documents(documents=documents, embedding=HuggingFaceEmbeddings())
#     retriever = vectorstore.as_retriever()
#     return retriever

def create_prompt():
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
             MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return prompt

def create_ragchain(retriever,model,prompt):
    question_answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()  # Initialize new session history
    return store[session_id]

def create_chat_hist(rag_chain):
    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    return chain_with_history

def create_gui(chain_with_history):
    st.set_page_config(page_title="Chatbot", page_icon="💬")
    st.title("PDF Question-Answering Assistant")
    st.write('Allows users to interact with the an AI')
    prompt = st.chat_input(placeholder="Ask Me Anything")
    if prompt:
        human_message = st.chat_message("human")
        human_message.write(f"USER:{prompt}")
        answer =  chain_with_history.invoke({"input": prompt},config={"configurable": {"session_id": "foo"}})['answer']
        ai_message = st.chat_message("assistant")
        ai_message.write(f"AI:{answer}")

        for message in store["foo"].messages:
            if isinstance(message, AIMessage):
                prefix = "AI"
            else:
                prefix = "User"

            print(f"{prefix}: {message.content}\n")

def main():
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    set_api("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")
    file_paths = [ r"C:\Users\parma\OneDrive\Documents\sanjana's documents\TYBSC\WD\Unit 4-Database and Email Handling.pdf",
                r"C:\Users\parma\OneDrive\Documents\sanjana's documents\TYBSC\WD\Unit 3-Arrays and File.pdf",
                r"C:\Users\parma\OneDrive\Documents\sanjana's documents\TYBSC\WD\Unit 2-Function and String.pdf",
                ]
    # all_splits = get_pdf(file_paths)
    retriever = create_vector_store_retriver(file_paths)

    system_prompt = create_prompt()
    model = ChatGroq(model="llama3-8b-8192")
    rag_chain = create_ragchain(retriever, model, system_prompt)
    chain_with_history = create_chat_hist(rag_chain)

    create_gui(chain_with_history)

    # while True:
    #     question = input("Enter question:")
    #     if question == "exit":
    #         break
    #     response = chain_with_history.invoke({"input": question},
    #                                         config={"configurable": {"session_id": "foo"}})['answer']
    #     print(response)
    #     for message in store["foo"].messages:
    #         if isinstance(message, AIMessage):
    #             prefix = "AI"
    #         else:
    #             prefix = "User"

    #         print(f"{prefix}: {message.content}\n")

if __name__ == "__main__":
    main()

