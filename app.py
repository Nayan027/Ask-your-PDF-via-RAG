from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain_perplexity import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

llm = ChatPerplexity()

prompt_template = ChatPromptTemplate.from_template(
    """Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.

In case a question comes out context, respond with: Sorry, out of context. Cannot answer this question.

    <context>
    {context}
    <context>
    
    Questions:{input}
"""
)

def clear_session():
    """Delete all keys from Streamlit session state"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]



def create_doc_embeddings():

    if "vector_store" not in st.session_state:

        st.session_state.doc_loader = PyPDFLoader("ISLP_website.pdf")
        st.session_state.documents = st.session_state.doc_loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        st.session_state.chunked_docs = st.session_state.text_splitter.split_documents(st.session_state.documents[:50])

        st.session_state.embedding_model = OllamaEmbeddings(model="nomic-embed-text")
        st.session_state.vector_store = FAISS.from_documents(documents=st.session_state.chunked_docs, embedding=st.session_state.embedding_model)




st.title("Chat with your PDF")

# Example usage (add a button to clear session manually)
if st.button("Clear Session"):
    clear_session()
    st.success("Session cleared!")


if st.button("Create Document Embeddings"):
    create_doc_embeddings()
    st.write("Vector-store DB is ready.")



user_input = st.text_input("Ask something")

if user_input:

    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)
    retriever = st.session_state.vector_store.as_retriever()

    retrieval_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)

    response = retrieval_chain.invoke({"input":user_input})
    st.write(response['answer'])


