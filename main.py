import streamlit as st
import time

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader


# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Document Q&A with Groq")


# -------------------- GROQ --------------------

try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error("GROQ_API_KEY not found in Streamlit Secrets.")
    st.stop()

llm = ChatGroq(
    api_key=groq_api_key,
    model="llama3-8b-8192",
    temperature=0
)


# -------------------- PROMPT --------------------

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question only from the provided context.

    <context>
    {context}
    </context>

    Question: {input}

    Provide a clear and concise answer.
    """
)


# -------------------- VECTOR DB --------------------

def create_vector_embedding():

    if "vectors" not in st.session_state:

        with st.spinner("Creating Vector Database..."):

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


# -------------------- BUTTON --------------------

if st.button("Create Vector Database"):
    create_vector_embedding()


# -------------------- QUERY INPUT --------------------

user_prompt = st.text_input(
    "Ask a question from your research papers"
)


# -------------------- RETRIEVAL --------------------

if user_prompt:

    if "vectors" not in st.session_state:
        st.warning("Please create the Vector Database first.")
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

    response_time = time.process_time() - start

    st.subheader("Answer")
    st.write(response["answer"])

    st.caption(f"Response Time: {response_time:.2f} seconds")

    with st.expander("Document Similarity Search"):

        for i, doc in enumerate(response["context"]):

            st.markdown(f"### Chunk {i+1}")
            st.write(doc.page_content)
            st.divider()
