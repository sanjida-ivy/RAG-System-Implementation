import json
from config import OLLAMA_MODEL, OLLAMA_HOST
from langchain_ollama.llms import OllamaLLM
from langchain.agents.react.agent import create_react_agent
from langchain.tools import Tool
from langchain import PromptTemplate
from utils import retrieve_docs, summarize_docs, answer_question
from pdf_processing import process_pdf

# Adapter for multi-step QA Calls
def answer_question_single(input_str: str) -> str:
    data = json.loads(input_str)
    return answer_question(query=data["query"], context=data["context"])

# Initialize Ollama LLM (HTTP)
llm = OllamaLLM(model=OLLAMA_MODEL, host=OLLAMA_HOST)

# Define tools
tools = [
    Tool.from_function(retrieve_docs,       name="RetrieveDocs",   description="Retrieve docs for a query."),
    Tool.from_function(summarize_docs,      name="SummarizeDocs",  description="Summarize docs."),
    Tool.from_function(answer_question_single,
                         name="AnswerQuestion",
                         description="Input JSON {'query','context'}."),
]

# Built-in React prompt
prompt = PromptTemplate(
    template="""
You are a helpful AI assistant with access to these tools:
{tools}

Use this format:

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

# Create the agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)