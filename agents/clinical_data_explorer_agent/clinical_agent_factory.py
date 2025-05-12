# clinical_agent_factory.py

# Agent Type: Clinical Data Explorer Agent
# Purpose: Explore EHR, lab data, clinical trials in CSV/Excel format
# Tools to Add: Pandas tool, PythonREPL, plot generator
# Use Case Example: “Plot CRP levels over time for diabetic patients.”
import matplotlib.pyplot as plt
import os
import uuid
import pandas as pd
from langchain_ollama.llms import OllamaLLM
from langchain.agents.react.agent import create_react_agent
from langchain.tools import Tool
from langchain import PromptTemplate
from config import OLLAMA_MODEL, OLLAMA_HOST
from . import clinical_tools



# ✅ Fallback REPL setup (manual if import fails)
class BasicPythonREPL:
    def run(self, command: str) -> str:
        try:
            local_vars = {}
            exec(command, globals(), local_vars)
            return str(local_vars) if local_vars else "Executed."
        except Exception as e:
            return f"Error: {e}"

llm = OllamaLLM(model=OLLAMA_MODEL, host=OLLAMA_HOST)

# Manual REPL tool
python_repl = BasicPythonREPL()
python_repl_tool = Tool(
    name="python_repl",
    func=python_repl.run,
    description="Run Python code on the loaded DataFrame `df`. Use print() to see output."
)

# Enhanced plot function
def enhanced_plot_variable(input_str: str) -> str:
    """
    Input format: 'file_path, column_name, [filter_column=condition]'
    Example: 'data.csv, CRP, Diagnosis=diabetes'
    """


    try:
        parts = [s.strip() for s in input_str.split(",")]
        file_path, column_name = parts[0], parts[1]
        filter_clause = parts[2] if len(parts) > 2 else None

        df = pd.read_csv(file_path)

        if filter_clause and "=" in filter_clause:
            filter_col, condition = [s.strip() for s in filter_clause.split("=")]
            df = df[df[filter_col].str.lower().str.contains(condition.lower())]

        plt.figure()
        df[column_name].reset_index(drop=True).plot(title=f"{column_name} Plot")
        plt.xlabel("Index")
        plt.ylabel(column_name)

        filename = f"plot_{uuid.uuid4().hex}.png"
        full_path = os.path.join("uploads", filename)
        plt.savefig(full_path)
        plt.close()

        return f"✅ Plot saved to {full_path}"
    except Exception as e:
        return f"❌ Failed to generate plot: {e}"

# Add your clinical tools
tools = [
    Tool.from_function(clinical_tools.load_csv_data, name="LoadCSV", description="Load and preview clinical CSV data. Input: file path"),
    Tool.from_function(
    enhanced_plot_variable,
    name="PlotVariable",
    description="Plot a column from clinical CSV. Input: file_path, column_name, [filter_column=condition] — e.g., 'data.csv, CRP, Diagnosis=diabetes'"
),
    python_repl_tool
]

prompt = PromptTemplate(
    template="""
You are a clinical data assistant...
...
{agent_scratchpad}
""",
    input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
)

clinical_agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

