# CodeBot: A Local LLM for Python Code Generation

## Overview

CodeBot is a Python application that utilizes a local language model to generate and check Python code based on user inputs. It integrates with Streamlit for an interactive UI and leverages OpenAI's API for language model interactions. This README provides an overview of the components and usage of CodeBot.

## Features

- Generates Python code based on user prompts.
- Checks the generated code for correctness.
- Provides an interactive UI using Streamlit.
- Uses OpenAI's language models for natural language processing.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/codebot.git
    cd codebot
    ```

2. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up environment variables**:
    Replace `sk-proj-xyz` with your actual OpenAI API key.
    ```python
    os.environ['OPENAI_API_KEY'] = 'sk-proj-xyz'
    ```

## Usage

1. **Run Streamlit**:
    ```bash
    streamlit run app.py
    ```

2. **Interact with the UI**:
    - Open your browser and navigate to `http://localhost:8501`.
    - Enter your prompt in the input field to get Python code suggestions.

## Code Structure

### Key Components

- **Prompt Template**:
    A structured prompt template is used to generate code solutions.
    ```python
    code_gen_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a coding assistant. ..."),
        ("placeholder", "{messages}")
    ])
    ```

- **CodeOutput Data Model**:
    A data model to structure the code output.
    ```python
    class CodeOutput(BaseModel):
        prefix: str
        imports: str
        code: str
    ```

- **Graph State**:
    Defines the state of the application for the state graph.
    ```python
    class GraphState(TypedDict):
        error: str
        messages: Annotated[List[AnyMessage], add_messages]
        generation: CodeOutput
        iterations: int
    ```

- **Functions**:
    - `generate(state: GraphState)`: Generates a code solution.
    - `code_check(state: GraphState)`: Checks the correctness of the generated code.
    - `decide_to_finish(state: GraphState)`: Decides whether to finish or retry based on the state.

- **Graph Initialization**:
    Initializes and builds the state graph.
    ```python
    def initialize_graph():
        ...
    ```

- **Streamlit Integration**:
    Provides a sidebar and main interface for user interaction.
    ```python
    with st.sidebar:
        ...
    st.title("💬 CodeBot")
    ...
    ```

## Example

1. **Enter a prompt**: "Generate a Python function to calculate the factorial of a number."
2. **CodeBot generates and checks the code**.
3. **View the generated code in the Streamlit interface**.

## Troubleshooting

- Ensure that LM Studio is running locally on `localhost:1234`.
- If the code generation fails, check the OpenAI API key and the connection to the local LLM.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
