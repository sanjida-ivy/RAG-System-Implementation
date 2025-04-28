from langchain.agents import initialize_agent, Tool, AgentType
from langchain.llms import Ollama  # Old import path
from utils import retrieve_docs, summarize_docs, answer_question

llm = Ollama(model="llama3")

tools = [
    Tool(
        name="RetrieveDocs",
        func=lambda query: retrieve_docs(query),
        description="Retrieve relevant documents based on the query"
    ),
    Tool(
        name="SummarizeDocs",
        func=lambda docs: summarize_docs(docs),
        description="Summarize the retrieved documents"
    ),
    Tool(
        name="AnswerQuestion",
        func=lambda inputs: answer_question(inputs["query"], inputs["context"]),
        description="Answer questions based on the provided context. Input should be a dictionary with 'query' and 'context' keys."
    ),
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)