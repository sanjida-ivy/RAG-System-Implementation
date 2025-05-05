# clinical_tools_lib.py
import pandas as pd

def load_csv_data(file_path: str) -> str:
    try:
        df = pd.read_csv(file_path)
        globals()["df"] = df
        cols = df.columns.tolist()
        preview = df.head().to_markdown()
        return f"CSV loaded successfully. Columns: {cols}\n\nPreview:\n{preview}"
    except Exception as e:
        return f"Failed to load CSV: {e}"


def plot_variable(input_str: str) -> str:
    # Placeholder - actual plotting logic here
    file_path, column_name = [s.strip() for s in input_str.split(",", 1)]
    return f"Plotted '{column_name}' from {file_path}"
