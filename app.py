# ---------------------- app.py ----------------------
import streamlit as st
import sys
# Must be first Streamlit command
st.set_page_config(page_title="🧠 Multi-Agent Assistant", layout="centered")

import os
import uuid
from PyPDF2 import PdfReader
from utils import answer_query
from agents.common.memory import auto_learn_facts  # Adjusted import path to match the correct location
from agents.common.pdf_processing import process_pdf
import importlib.util
from agents.clinical_data_explorer_agent.clinical_agent_factory import clinical_agent

# Add the root directory of your project to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    st.markdown("<h1 style='text-align: center;'>🧠 Ask Your Assistant</h1>", unsafe_allow_html=True)

    selected_agent = st.selectbox("Select Agent", ["General Agent", "Clinical Data Explorer Agent"])

    if "history" not in st.session_state:
        st.session_state.history = []

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    st.markdown("<hr>", unsafe_allow_html=True)

    user_input = None
    uploaded_files = None
    uploaded_csv = None
    csv_path = None

    if selected_agent == "General Agent":
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input("Ask your question...", key="general_input")
        with col2:
            uploaded_files = st.file_uploader("📎 Upload PDFs", type="pdf", accept_multiple_files=True, label_visibility="collapsed")

    elif selected_agent == "Clinical Data Explorer Agent":
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input("Ask about your clinical CSV...", key="clinical_input")
        with col2:
            uploaded_csv = st.file_uploader("📊 Upload Clinical CSV", type="csv", label_visibility="collapsed")

    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.history.append({"role": "user", "content": user_input})

        response = ""
        all_text = ""

        if selected_agent == "General Agent":
            if uploaded_files:
                for f in uploaded_files:
                    file_path = os.path.join("uploads", f"{uuid.uuid4()}-{f.name}")
                    with open(file_path, "wb") as out:
                        out.write(f.read())
                    reader = PdfReader(file_path)
                    full_text = "".join(page.extract_text() or "" for page in reader.pages)
                    all_text += f"\n--- From {f.name} ---\n{full_text}"
                    st.session_state["pdf_raw_text"] = all_text
                    count = process_pdf(file_path)
                    st.success(f"✅ Indexed {count} chunks from {f.name}")

                preview = all_text[:3000] + ("..." if len(all_text) > 3000 else "")
                with st.chat_message("assistant"):
                    with st.expander("📄 PDF content uploaded (click to view)", expanded=False):
                        st.code(preview)
                st.session_state.history.append({
                    "role": "assistant",
                    "content": f"📄 PDF content uploaded:\n\n```\n{preview}\n```"
                })

            with st.spinner("Thinking..."):
                response = answer_query(user_input)

        elif selected_agent == "Clinical Data Explorer Agent":
            if uploaded_csv:
                csv_path = os.path.join("uploads", f"{uuid.uuid4()}-{uploaded_csv.name}")
                with open(csv_path, "wb") as out:
                    out.write(uploaded_csv.read())
                st.session_state["csv_path"] = csv_path

                with st.spinner("Thinking..."):
                    
                    input_data = {
                        "input": f"{csv_path} | {user_input}",
                        "intermediate_steps": []
                    }
                    response = clinical_agent.invoke(input_data)
            else:
                st.warning("Please upload a clinical CSV file.")

        if response:
            auto_learn_facts(response, user_id="default")
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()