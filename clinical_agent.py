# clinical_agent.py

from langchain_ollama.llms import OllamaLLM
from langchain.agents.react.agent import create_react_agent
from langchain.tools import Tool
from langchain import PromptTemplate
from config import OLLAMA_MODEL, OLLAMA_HOST
import clinical_tools

# LLM setup
llm = OllamaLLM(model=OLLAMA_MODEL, host=OLLAMA_HOST)

# Define tools
load_tool = Tool.from_function(
    func=clinical_tools.load_csv_data,
    name="LoadCSV",
    description="Load and preview clinical CSV data. Input: file path"
)

plot_tool = Tool.from_function(
    func=clinical_tools.plot_variable,
    name="PlotVariable",
    description="Plot a column from clinical CSV. Input: file path and column name separated by comma"
)

tools = [load_tool, plot_tool]

# Prompt for Clinical Agent
prompt = PromptTemplate(
    template="""
You are a clinical data assistant. Use the tools to explore and visualize clinical study CSV data.

{tools}

Question: {input}
Thought: think
Action: one of [{tool_names}]
Action Input: the input for that tool
Observation: the tool output
... repeat ...
Final Answer: your answer

{agent_scratchpad}
""",
    input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
)

# Agent creation
clinical_agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)
