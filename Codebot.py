# This is a streamlit app! Watchout! See README
# CLI% streamlit run pyrrito.py
# filename: pyrrito.py
# author: matrodge
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
from streamlit import util


# truncation limit
TRUNCATE = 100000

import functools

# Logging decorator
def log_state_change(func):
    @functools.wraps(func)
    def wrapper_log_state_change(*args, **kwargs):
        state = args[0] if args else {}
        print(f"State before '{func.__name__}': {state}")
        result = func(*args, **kwargs)
        print(f"State after '{func.__name__}': {result}")
        return result
    return wrapper_log_state_change

# because I use OpenAI langchain, I am forced to use OpenAI API key here...
os.environ['OPENAI_API_KEY'] = 'just_kidding' 

# Initialize OpenAI LLM
llm = ChatOpenAI(model="llama3", temperature=0)

# Prompt template for generating code
code_gen_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert coding assistant. You ALWAYS PROVIDE COMPLETE CODE. Ensure all code you provide is complete and can be executed with all required imports and variables defined. Structure your answer: 1) a prefix describing the code solution, 2) the imports, 3) the fully functional code. Here is the user request:"),
    ("user", "{messages}")
])

# Data model for code output
class CodeOutput(BaseModel):
    """Schema for code solution to the requests."""
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code import statements")
    code: str = Field(description="Code not including import statements")

# Configure LLM with structured output
code_gen_chain = llm.with_structured_output(CodeOutput, include_raw=False)

# Define the state of our graph
class GraphState(TypedDict):
    error: str
    messages: Annotated[List[AnyMessage], add_messages]
    generation: CodeOutput
    iterations: int

# Function to generate a code solution
@log_state_change
def generate(state: GraphState):
    print("---GENERATING CODE SOLUTION---")
    messages, iterations = state["messages"], state["iterations"]
    code_solution = code_gen_chain.invoke(messages)
    #print(f"in Generate response: {code_solution}")
    messages.append(("assistant", f"Here is my attempt to solve the coding problem: {code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}"))
    return {"generation": code_solution, "messages": messages, "iterations": iterations + 1}

# Function to check code correctness
@log_state_change
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
@log_state_change
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
    "PYRITTO - It's like Chipotle for Python."

    "[Get LM Studio Set Up](https://lmstudio.ai/)"

    "Ensure LM Studio is running locally on localhost:1234."

    "I truncate code over 100,000 characters long... sorry."

    """Example Prompt:
    
    Write a 'basic_data_analysis.py' Python script that:
    - Accepts a CSV file path as input from the command line.
    - Calculates the sum, average, minimum, and maximum of a given numerical column.
    - Counts the unique values in a given categorical column.
    - Generates and saves a histogram or bar chart for the specified column.

    Ensure the script:
    - Utilizes pandas for data operations and matplotlib for visualizations.
    - Completes tasks within a single function, running efficiently on datasets up to 1,000 rows.
    - Focuses solely on the specified tasks without extra features.
    - Gracefully handles errors with clear messages, such as for missing files or columns.
    The script should be straightforward, avoiding over-complexity, and execute in under one minute for the intended data size."""


st.title("🌯 Pyritto")
st.caption("🚀 v0.1 tasty, quick and dirty python, while you watch behind the glass... 🚀")

def reset_conversation():
  st.session_state.conversation = None
  st.session_state.chat_history = None
st.button('Reset Chat', on_click=reset_conversation)




if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi, welcome to Pyritto! What can I get started for you today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    client = openai.OpenAI(base_url="http://localhost:1234/v1/")
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
    except ConnectionError:
        # Handle network-related errors when streaming data from LM Studio's API
        util.info("Failed to connect to the Codex service due to a network issue.")
    except Exception as e:
        # Catch-all for any unforeseen exceptions that may occur during the streaming process
        util.info(f"A general exception was caught: {e}")
        st.error("An unknown error has occurred. We're working on fixing it!")

    # Get response from local LLM
    response = client.chat.completions.create(model="gpt-4-turbo", messages=st.session_state.messages)
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
