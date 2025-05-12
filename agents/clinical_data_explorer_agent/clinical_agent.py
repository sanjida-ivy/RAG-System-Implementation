from langchain.agents import Tool
from langchain.agents import create_react_agent
from langchain_ollama.llms import OllamaLLM
from agents.clinical_data_explorer_agent.clinical_tools import load_csv_data

# Set up LLM
llm = OllamaLLM(model=OLLAMA_MODEL, host=OLLAMA_HOST)

# Define tools
tools = [
    Tool.from_function(
        func=load_csv_data,
        name="LoadCSV",
        description="Load and preview clinical CSV data. Input: file path"
    )
]

# Create agent
clinical_agent = create_react_agent(
    llm=llm,
    tools=tools  # ✅ REQUIRED argument
    # Optional: prompt=custom_prompt
)
