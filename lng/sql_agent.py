# LangChain AI agent.
#
# An Agent implementing a native language queries using LangGraph / LangChain framework.
# It is probably the oldest, but rather low-level forcing to use some extra utility logic.
#
# Run it with streamlit:
#
#     streamlit run lng/app.py --server.address 127.0.0.1 --server.port 7932
#
# Either local Ollama or cloud-based Deepseek / YandexGPT LLM's could be used.
# Model selection is fixed (see `MODEL_TYPE`).
#

import os
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from typing import cast, Literal

ModelType = Literal['ollama', 'deepseek', 'yandex']
""" Allowed model types. Change `MODEL_TYPE` below.  """

# Constants
DB_FILENAME = './data/titanic.db'
""" SQLite database """

MODEL_TYPE: ModelType = 'yandex'
""" Model type selection """

# Setup the model
def get_model(model_type: ModelType) -> BaseChatModel:
    """ Return Chat LLM instance """

    match model_type:
        case 'ollama':
            # Local Ollama instance provider
            return ChatOllama(model=os.environ.get('OLLAMA_MODEL', 'gpt-oss:20b'))

        case 'deepseek' if 'DEEPSEEK_API_KEY' in os.environ:
            # Deepseek provider
            return ChatDeepSeek(
                model=os.environ.get('DEEPSEEK_MODEL', 'deepseek-chat'),
                api_key=os.environ['DEEPSEEK_API_KEY'], # type:ignore
            )

        case 'yandex' if 'YANDEX_API_KEY' in os.environ and 'YANDEX_FOLDER_ID' in os.environ:
            # YandexGPT provider
            from yandex_ai_studio_sdk import AIStudio
            from yandex_ai_studio_sdk.auth import APIKeyAuth

            sdk = AIStudio(folder_id=os.environ['YANDEX_FOLDER_ID'], auth=APIKeyAuth(os.environ['YANDEX_API_KEY']))
            model = sdk.models.completions(os.environ.get('YANDEX_MODEL', 'yandexgpt')).langchain()
            return cast(BaseChatModel, model)

        case _:
            raise ValueError(f'Unknown or unconfigured model {model_type}')

model = get_model('ollama')
""" LLM instance """

# Setup the database
db_file = os.path.abspath(DB_FILENAME)
assert os.path.exists(db_file), f'Database {db_file} does not exist'

db = SQLDatabase.from_uri(f"sqlite:///{db_file}?mode=ro", engine_args={"echo": True})
""" Database instance """

SYSTEM_PROMPT = '''
## SQLite Database Toolset - Instructions

You are an database analyst with access to SQLite database tools.
Your task is to answer to user questions by retrieving data from database while strictly adhering to the rules and procedures outlined below.

### Available tools for database operations and querying:

* `sql_db_list_tables()` - returns a list of all tables in the database.
* `sql_db_schema(table_name: str)` - returns detailed structure of the specified table: columns, data types, NULL/NOT NULL, primary key, foreign keys, indexes.
* `sql_db_query_checker(sql: str)` - verify the query, returns error if query is invalid.
* `sql_db_query(sql: str)` - executes an SQL query (SELECT only) and returns the data.

### Constraints
* **SELECT only**. No INSERT, UPDATE, DELETE, CREATE, ALTER, DROP.
* **SQLite functions only** - use only functions supported by SQLite.
* **Always validate the schema** before constructing queries with JOIN.
* **Never guess** about relationships between tables, rely solely on schema information.
* Do not ask for permission to execute a query, run it immediately (if you are confident it is correct).
* Do not include the SQL in the response, return only the data and, if needed, explanations.
* If `describe_table` returns an error - stop and report the problem (table not found).

### Schema Validation
If the user requests data that requires joining tables, follow these steps strictly:

* Determine which tables are likely needed (based on the request).
* For each such table mandatorily call describe_table (e.g., describe_table({"table_name": "orders"})).
* Locate foreign keys (foreign_keys) in the table description.
* Check whether a foreign key exists linking these tables (one table references the primary key of the other).
* If an explicit FK exists - use it as the join condition (JOIN … ON …).
* If no FK is declared – DO NOT perform the JOIN. Instead, respond:
    “I could not find a foreign key between tables <A> and <B>. Please specify the joining columns.”
And invite them to provide the condition manually.

### Sample Workflow
* User: “Show me all orders that belong to the product ‘Chai’.”
* Assistant
    * describe_table → orders – confirm FK to order_details via OrderID.
    * describe_table → order_details – confirm FK to products via ProductID.
    * describe_table → products – confirm FK to categories.
    * Build a JOIN chain: orders → order_details → products.
    * Return query results.
'''
''' The system prompt '''

# Setup database toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=model)
toolset = toolkit.get_tools()

# Graph utility functions
def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

def should_use_tools(state: MessagesState) -> str:
    messages = state['messages']
    last_message = messages[-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls: # type: ignore
        return "tools"

    return END

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

# Setup the graph
model = model.bind_tools(toolset)
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", 
                  ToolNode(toolset).with_fallbacks(
                       [RunnableLambda(handle_tool_error)], exception_key="error"))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges('agent', should_use_tools)
workflow.add_edge("tools", 'agent')
app = workflow.compile(checkpointer=MemorySaver(), debug=False)

# def run_and_print(messages):
#     for chunk in app.stream(
#         {"messages": messages}, 
#         stream_mode="values",
#         config={"configurable": {"thread_id": 42}}):
#             chunk["messages"][-1].pretty_print()
#             messages.extend(chunk["messages"])
#     return messages
# 
# messages = run_and_print([
#     ("system", SYSTEM_PROMPT),
#     ("human", "What tables are available in Titanic database?")
# ])
#
# messages = run_and_print(messages + [("human", "How many people are in total in Titanic database?")])
# messages = run_and_print(messages + [('human', 'How many of them survived?')])
# messages = run_and_print(messages + [('human', 'Generate a single SQL query to get numeric facts provided in previous answers')])
