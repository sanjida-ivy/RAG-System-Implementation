import streamlit as st
from agent import agent
from utils import process_pdf
import uuid
import os

st.set_page_config(page_title="RAG System", layout="wide")
st.title("🧠 Agentic RAG System with LangChain + Pinecone + Ollama")

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload a PDF", type="pdf")
pdf_filename = None

if uploaded_file:
    pdf_filename = f"uploaded_{uuid.uuid4().hex[:8]}.pdf"
    with open(pdf_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("🔍 Extracting and embedding PDF..."):
        try:
            text, embedding = process_pdf(pdf_filename)
            st.success("✅ PDF processed successfully!")
            st.text_area("📝 Extracted Text", value=text[:2000], height=300)
        except Exception as e:
            st.error(f"🚫 Error processing PDF: {e}")
        finally:
            if pdf_filename and os.path.exists(pdf_filename):
                os.remove(pdf_filename)

# User query input
query = st.text_input("💬 Ask a question about the uploaded document:")
if query:
    with st.spinner("🤖 Thinking..."):
        try:
            response = agent.run(query)
            st.markdown("### 🧠 Final Answer:")
            st.write(response)
        except Exception as e:
            st.error(f"🚫 Agent failed to respond: {e}")