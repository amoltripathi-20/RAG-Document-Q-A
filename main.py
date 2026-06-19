import os
import time
import streamlit as st

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader


# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit page config
st.set_page_config(page_title="RAG PDF Q&A", page_icon="📚")
st.title("📚 RAG Document Q&A with Groq")

# LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-8b-8192"
)

# Prompt
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the provided context.

    <context>
    {context}
    </context>

    Question: {input}
    """
)


def create_vector_embedding():

    if "vectors" not in st.session_state:

        with st.spinner("Loading PDFs and creating embeddings..."):

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            loader = PyPDFDirectoryLoader("research_papers")

            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            final_documents = text_splitter.split_documents(docs)

            vectors = FAISS.from_documents(
                final_documents,
                embeddings
            )

            st.session_state.vectors = vectors

        st.success("Vector Database Created Successfully!")


# Button
if st.button("Create Vector Database"):
    create_vector_embedding()

# Query box
user_prompt = st.text_input(
    "Ask a question from your research papers:"
)

# Retrieval
if user_prompt:

    if "vectors" not in st.session_state:
        st.warning("Please create the vector database first.")
        st.stop()

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    retriever = st.session_state.vectors.as_retriever()

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    start = time.process_time()

    response = retrieval_chain.invoke(
        {"input": user_prompt}
    )

    st.write("### Answer")
    st.write(response["answer"])

    st.write(
        f"Response Time: {time.process_time() - start:.2f} seconds"
    )

    with st.expander("Retrieved Chunks"):

        for i, doc in enumerate(response["context"]):
            st.write(f"Chunk {i+1}")
            st.write(doc.page_content)
            st.write("---")
