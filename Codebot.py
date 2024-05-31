import os
import textwrap
import uuid
from datetime import datetime
from typing import List, TypedDict, Annotated
import openai
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

# truncation limit
TRUNCATE = 100000

# because I use OpenAI langchain, I am forced to use OpenAI API key here...
os.environ['OPENAI_API_KEY'] = 'sk-proj-xyz'  # Replace with your actual key or use a more secure method

# Initialize OpenAI LLM
llm = ChatOpenAI(model="llama3", temperature=0)

# Prompt template for generating code
code_gen_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a coding assistant. Ensure any code you provide can be executed with all required imports and variables defined. Structure your answer: 1) a prefix describing the code solution, 2) the imports, 3) the functioning code block. Here is the user question:"),
    ("placeholder", "{messages}")
])

# Data model for code output
class CodeOutput(BaseModel):
    """Schema for code solutions to questions."""
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")

# Configure LLM with structured output
code_gen_chain = llm.with_structured_output(CodeOutput, include_raw=False)

# Define the state of our graph
class GraphState(TypedDict):
    error: str
    messages: Annotated[List[AnyMessage], add_messages]
    generation: CodeOutput
    iterations: int

# Function to generate a code solution
def generate(state: GraphState):
    print("---GENERATING CODE SOLUTION---")
    messages, iterations = state["messages"], state["iterations"]
    code_solution = code_gen_chain.invoke(messages)
    print(f"in Generate response: {code_solution}")
    messages.append(("assistant", f"Here is my attempt to solve the problem: {code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}"))
    return {"generation": code_solution, "messages": messages, "iterations": iterations + 1}

# Function to check code correctness
def code_check(state: GraphState):
    print("---CHECKING CODE---")
    messages, code_solution = state["messages"], state["generation"]
    imports, code = code_solution.imports, code_solution.code

    # Check imports
    try:
        local_scope = {}
        exec(imports, {}, local_scope)
    except Exception as e:
        error_message = f"Your solution failed the import test. Error: {e}."
        messages.append(("user", error_message))
        return {"generation": code_solution, "messages": messages, "iterations": state["iterations"], "error": "yes"}

    # Check code execution
    try:
        combined_code = f"{imports}\n{code}"
        exec(combined_code, {}, local_scope)
    except Exception as e:
        error_message = f"Your solution failed the code execution test. Error: {e}."
        messages.append(("user", error_message))
        return {"generation": code_solution, "messages": messages, "iterations": state["iterations"], "error": "yes"}

    # If successful, print the nicely formatted code
    print("\n---CODE EXECUTION SUCCESSFUL---\n")
    dedented_code = textwrap.dedent(combined_code)
    print(dedented_code)

    return {"generation": code_solution, "messages": messages, "iterations": state["iterations"], "error": "no"}

max_iterations = 3

# Function to decide whether to finish or retry
def decide_to_finish(state: GraphState):
    if state["error"] == "no" or state["iterations"] == max_iterations:
        return "end"
    return "generate"

# Initialize and build the graph
def initialize_graph():
    builder = StateGraph(GraphState)
    builder.add_node("generate", generate)
    builder.add_node("check_code", code_check)
    builder.set_entry_point("generate")
    builder.add_edge("generate", "check_code")
    builder.add_conditional_edges("check_code", decide_to_finish, {"end": END, "generate": "generate"})

    memory = SqliteSaver.from_conn_string(":memory:")
    graph = builder.compile(checkpointer=memory)
    graph.get_graph().print_ascii()
    return graph

# Utility function to print events
def _print_event(event: dict, _printed: set, max_length=TRUNCATE):
    current_state = event.get("dialog_state")
    if current_state:
        print(f"Currently in: {current_state[-1]}")
    
    message = event.get("messages")
    if message:
        if isinstance(message, list) and message:
            message = message[-1]
        
        # Check if message is a tuple and unpack it
        if isinstance(message, tuple):
            _, message_content = message
            msg_repr = message_content if isinstance(message_content, str) else str(message_content)
            message_id = id(message_content)  # Use id() to get a unique identifier for the content
        else:
            msg_repr = str(message)
            message_id = id(message)  # Use id() to get a unique identifier
        
        # Truncate if necessary
        if len(msg_repr) > max_length:
            msg_repr = msg_repr[:max_length] + " ... (truncated)"
        
        # Print and add to printed set
        if message_id not in _printed:
            print(msg_repr)
            _printed.add(message_id)

# Streamlit UI Integration
with st.sidebar:
    #openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get LM Studio Set Up](https://lmstudio.ai/)"
    "Make sure LM Studio is running locally on localhost:1234"
    "I truncate code over 100000 characters long...sorry."
  
openai_api_key = "none"

st.title("💬 CodeBot")
st.caption("🚀 v0.1 a Local LLM to get some quick and dirty python written...")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "What you need?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please type any string to continue.")
        st.stop()

    client = openai.OpenAI(api_key=openai_api_key, base_url="http://localhost:1234/v1/")
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Code generation graph integration
    graph = initialize_graph()
    state = {"messages": [("user", prompt)], "iterations": 0}
    _printed = set()
    config = {"thread_id": str(uuid.uuid4()), "thread_ts": datetime.now().isoformat()}
    
    # Debug statements for thread_id and thread_ts
    print(f"Thread ID: {config['thread_id']}")
    print(f"Thread TS: {config['thread_ts']}")

    # Debug the state and config before streaming
    print(f"State: {state}")
    print(f"Config: {config}")

    try:
        events = graph.stream(state, config, stream_mode="values")
        for event in events:
            _print_event(event, _printed)
            if "generation" in event:
                code_solution = event["generation"]
                dedented_code = textwrap.dedent(f"{code_solution.imports}\n{code_solution.code}")
                st.code(dedented_code, language='python')
    except ValueError as e:
        print(f"ValueError: {e}")
        print("Ensure 'thread_id' and 'thread_ts' are correctly passed to the config.")
        print(f"Config passed: {config}")

    # Get response from local LLM
    response = client.chat.completions.create(model="gpt-4-turbo", messages=st.session_state.messages)
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
