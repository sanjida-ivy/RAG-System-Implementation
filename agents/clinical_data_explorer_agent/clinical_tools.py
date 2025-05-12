# ---------------- clinical_tools.py ----------------
import os
# Removed the circular import: from clinical_agent import clinical_agent

os.makedirs("uploads", exist_ok=True)

# Utility functions like load_csv_data, plot_variable would live here or better: in clinical_tools_lib.py
# No top-level Streamlit commands like st.set_page_config here!
