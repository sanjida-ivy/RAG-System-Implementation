# ---------------- clinical_tools.py ----------------
import os
import uuid
import streamlit as st
from PyPDF2 import PdfReader

from pdf_processing import process_pdf
from utils import answer_query
from memory import auto_learn_facts
# Removed the circular import: from clinical_agent import clinical_agent

os.makedirs("uploads", exist_ok=True)

# Utility functions like load_csv_data, plot_variable would live here or better: in clinical_tools_lib.py
# No top-level Streamlit commands like st.set_page_config here!
