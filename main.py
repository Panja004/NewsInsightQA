import os
import streamlit as st
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline


st.title("NewsInsightQA: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_hf.pkl"

main_placeholder = st.empty()

def build_vector_store(urls, file_path):
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅")
    data = loader.load()
    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    main_placeholder.text("Text Splitting...Started...✅")
    docs = text_splitter.split_documents(data)
    # Create embeddings and save to FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore_hf = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Building...✅")
    time.sleep(1)
    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_hf, f)

if process_url_clicked and urls:
    build_vector_store(urls, file_path)
    st.sidebar.success("Processing complete! You can now ask questions.")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            # Load local HuggingFace pipeline
            qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_length=512)
            llm = HuggingFacePipeline(pipeline=qa_pipeline)
            combine_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce")
            chain = RetrievalQAWithSourcesChain(
                combine_documents_chain=combine_chain,
                retriever=vectorstore.as_retriever(),
                return_source_documents=True
            )
            response = chain({"question": query})
            st.header("Answer")
            st.write(response["answer"])

            # Display sources
            sources = set(doc.metadata.get('source', 'Unknown') for doc in response['source_documents'])
            if sources:
                st.subheader("Sources:")
                for source in sources:
                    st.write(source)
    else:
        st.warning("Please process URLs first.")

