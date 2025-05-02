import os
import uuid
import streamlit as st
from PyPDF2 import PdfReader

from pdf_processing import process_pdf
from utils import answer_query

st.set_page_config(page_title="🧠 Ask Anything", layout="centered")
os.makedirs("uploads", exist_ok=True)


def main():
    st.markdown("<h1 style='text-align: center;'>🧠 Ask Questions — With or Without PDFs</h1>", unsafe_allow_html=True)

    if "history" not in st.session_state:
        st.session_state.history = []

    # --- Chat Window ---
    chat_placeholder = st.container()
    with chat_placeholder:
        for msg in st.session_state.history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # --- Bottom Input Area with File Upload ---
    st.markdown("<hr>", unsafe_allow_html=True)
    with st.container():
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input("Type your question and hit Enter...", key="user_input")
        with col2:
            uploaded_files = st.file_uploader("📎", type="pdf", accept_multiple_files=True, label_visibility="collapsed")

    # --- Handle Submission ---
    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.history.append({"role": "user", "content": user_input})

        all_text = ""

        # ✅ Process uploaded PDFs before answering
        if uploaded_files:
            for f in uploaded_files:
                file_path = os.path.join("uploads", f"{uuid.uuid4()}-{f.name}")
                with open(file_path, "wb") as out:
                    out.write(f.read())

                # Extract raw text
                reader = PdfReader(file_path)
                full_text = "".join(page.extract_text() or "" for page in reader.pages)
                all_text += f"\n--- From {f.name} ---\n{full_text}"

                # Store for answer_query
                st.session_state["pdf_raw_text"] = all_text
                count = process_pdf(file_path)
                st.success(f"✅ Indexed {count} chunks from {f.name}")

            # 💬 Show extracted PDF text in assistant-style bubble (scrollable)
            preview = all_text[:3000] + ("..." if len(all_text) > 3000 else "")
            with st.chat_message("assistant"):
                with st.expander("📄 PDF content uploaded (click to view)", expanded=False):
                    st.code(preview)
            st.session_state.history.append({
                "role": "assistant",
                "content": f"📄 PDF content uploaded:\n\n```\n{preview}\n```"
            })

        # 🔍 Get Assistant Response
        with st.spinner("Thinking..."):
            response = answer_query(user_input)

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
