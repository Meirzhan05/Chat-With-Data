import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import bs4
from PyPDF2 import PdfReader


load_dotenv()

def get_text_from_url(url):
    web_loader = WebBaseLoader(url)
    document = web_loader.load()
    return document

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks_from_url(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    document_chunks = text_splitter.split_documents(text)
    return document_chunks

def get_text_chunks_from_pdf(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore_from_url(document_chunks):
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    embeddings = vector_store.embeddings

    st.write(embeddings)
    return vector_store

def get_vectorstore_from_pdf(text_chunks):
    vector_store = Chroma.from_texts(text_chunks, OpenAIEmbeddings())
    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
    
    retriever = vector_store.as_retriever(search_type="similarity")
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""


    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        contextualize_q_system_prompt
    ])
    

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

def get_conversational_rag_chain(retriever_chain):

    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    prompt = ChatPromptTemplate.from_messages([
      ("system", qa_system_prompt),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)

    conversational_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversational_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response['answer']

# app config
st.set_page_config(page_title="Chat With Website")

st.title("Chat With Website")

# sidebar
with st.sidebar:
    st.header("Settings")
    input_option = st.radio("Choose input method", ("Website URL", "PDF file"))
    if input_option == "Website URL":
        website_url = st.text_input("Website URL")
    else:
        uploaded_files = st.file_uploader("Upload a PDF file", type=['pdf'])
    process_btn = st.button("Process")

# main content
with st.spinner("Processing"):
    if input_option == "Website URL" and (website_url is None or website_url == ""):
        st.info("Please enter a website URL")
    elif input_option == "PDF file" and not process_btn and uploaded_files == []:
        st.info("Please upload a PDF file")
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "vector_store" not in st.session_state:
            if input_option == "Website URL" and website_url is not None:
                text = get_text_from_url(website_url)
                text_chunks = get_text_chunks_from_url(text)
                st.write(text_chunks)
                st.session_state.vector_store = get_vectorstore_from_url(text_chunks)
            elif input_option == "PDF file" and uploaded_files is not None:
                text = get_pdf_text(uploaded_files)
                text_chunks = get_text_chunks_from_pdf(text)
                st.session_state.vector_store = get_vectorstore_from_pdf(text_chunks)

        user_query = st.chat_input("Type a prompt...")

        if user_query is not None and user_query != "":
            response = get_response(user_query)
            
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))


            # chat
            for message in st.session_state.chat_history:
                if isinstance(message, AIMessage):
                    with st.chat_message("AI"):
                        st.write(message.content)
                elif isinstance(message, HumanMessage):
                    with st.chat_message("Human"):
                        st.write(message.content)