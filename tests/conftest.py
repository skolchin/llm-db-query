# PyTest fixtures

import os
import logging
import pytest
import logging
from pathlib import Path

from utils import stop_ollamas

_logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.ERROR)

# ("deepseek", "deepseek-reasoner") removed as it's not good with tool calls
# see https://github.com/langchain-ai/langchain/issues/34166
MODELS = [("ollama", "gpt-oss:20b"),
          ("ollama", "qwen3:30b"),
          ("deepseek", "deepseek-chat"),]

MODEL_IDS = [v[1] for v in MODELS]

PYAI_SERVERS = [("sql", "pyai/mcp_sql_server.py"),
                ("bsl", "pyai/mcp_bsl_server.py")]

PYAI_SERVER_IDS = [v[0] for v in PYAI_SERVERS]

@pytest.fixture(scope="session")
def num_perf_rounds(pytestconfig) -> int:
    """ Number of performance testing rounds.

    Set by `--benchmark-min-rounds` pytest option (command line or pytest.ini).
    Default is 30.
    """
    if not (sess := getattr(pytestconfig, "_benchmarksession", None)):
        raise ValueError("Pytest-benchmark is not available for this session")
    
    return sess.options.get('min_rounds', 30)

@pytest.fixture(scope='session')
def sqldb_filename():
    """Test database file (Northwind)"""
    return os.path.abspath('./data/northwind.db')

@pytest.fixture(scope='session')
def sqldb_database(sqldb_filename):
    """Test database instance (Northwind)"""
    from langchain_community.utilities.sql_database import SQLDatabase
    return SQLDatabase.from_uri(f"sqlite:///{sqldb_filename}?mode=ro", engine_args={"echo": False})

@pytest.fixture(params=MODELS, ids=MODEL_IDS)
def lng_agent_model(request):
    """LNG agent LLM instance (parametrized)"""
    from lng.get_model import get_model

    _logger.info(f"Running test with '{request.param[0]}:{request.param[1]}' agent")
    yield get_model(request.param[0], request.param[1])
    stop_ollamas(request.param[0])

@pytest.fixture(params=MODELS, ids=MODEL_IDS)
def lng_agent_model_qual_name(request):
    """LNG agent LLM qualified name (parametrized)"""
    from lng.get_model import get_model_qualified_name

    _logger.info(f"Running test with '{request.param[0]}:{request.param[1]}' agent")
    yield get_model_qualified_name(request.param[0], request.param[1])
    stop_ollamas(request.param[0])

@pytest.fixture(scope='session')
def lng_bsl_agent_prompt() -> str:
    """ BSL LNG agent prompt template """
    return """
Answer to user question: {question}.
Return ONLY valid JSON array.
NO explanations. NO comments.
Each array element MUST correspond to one row.
All column names MUST BE in CamelCase (for example CompanyName, OrderCount). No underscores allowed.
"""

@pytest.fixture(scope='session')
def lng_sql_agent_prompt() -> str:
    """ SQL LNG agent prompt """
    return """
You are a SQLite database query agent.

Answer user questions by retrieving data using SQL.

### Available tools

* `sql_db_list_tables()` - returns a list of all tables in the database.
* `sql_db_schema(table_name: str)` - returns detailed structure of the specified table: columns, data types, NULL/NOT NULL, primary key, foreign keys, indexes.
* `sql_db_query_checker(sql: str)` - verify the query, returns error if query is invalid.
* `sql_db_query(sql: str)` - executes an SQL query (SELECT only) and returns the data.

### Rules

- Only SELECT queries are allowed.
- Use only SQLite syntax.
- Never guess table names, column names, or relationships.
- Always inspect schema before writing a query.
- Always validate SQL using sql_db_query_checker before execution.
- If schema inspection fails, stop and report the error.
- Never show SQL queries in the final answer.
- Never explain anything.

### Column naming rules

- Every selected column must be aliased exactly as "TableName.ColumnName",
  where TableName is the name of table this column is in database.
  If the column is calculated, determine the table from the formula.
- Do not change names.
- Use exact CamelCase from schema.
- Examples:
    SELECT Customers.CompanyName AS "Customers.CompanyName" FROM Customers
    SELECT count() AS "Orders.OrderCount" FROM Orders

### Process

1. If tables are unknown → call sql_db_list_tables().
2. Inspect needed tables using sql_db_schema().
3. Write a SELECT query.
4. Validate with sql_db_query_checker().
5. Execute with sql_db_query().
6. Return the result ONLY as a valid JSON array, with no additional text, explanations, or commentary.

### Output format

- Return ONLY a valid JSON array.
- An array element must correspond to one data record.
- DO NOT INCLUDE:
-   explanations, 
-   SQL,
-   comments,
-   markdown.

The output must consist EXCLUSIVELY of the JSON array – no whitespace outside the array, no introductory phrases, no trailing sentences.
**Failure to follow the output format exactly (including any extra text) is considered an error.**

Example:

[
  {{
    "Customers.CompanyName": "ABC Corp",
    "Orders.OrderCount": 15
  }}
]
"""

@pytest.fixture
def lng_bsl_agent(lng_agent_model_qual_name):
    """ LNG BSL agent instance """
    from boring_semantic_layer.agents.backends.langgraph import LangGraphBackend

    agent = LangGraphBackend(
        model_path=Path('./data/northwind_bsl.yaml'),
        llm_model=lng_agent_model_qual_name,
        chart_backend="plotext",
        profile="northwind_duckdb",
        profile_file=Path("./data/northwind_profile.yaml"),
        return_json=True,
    )
    yield agent

@pytest.fixture
def lng_sql_agent(lng_agent_model, sqldb_database, lng_sql_agent_prompt):
    """ LNG SQL agent instance """
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
    from langchain_community.agent_toolkits.sql.base import create_sql_agent

    toolkit = SQLDatabaseToolkit(db=sqldb_database, llm=lng_agent_model)

    prompt = ChatPromptTemplate.from_messages([
        ("system", lng_sql_agent_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_sql_agent(
        top_k=1000,
        prompt=prompt,
        llm=lng_agent_model,
        toolkit=toolkit,
        agent_type="tool-calling",
        verbose=True,
    )
    return agent

@pytest.fixture(scope='session')
def pyai_agent_prompt() -> str:
    """ PyAI agent prompt template """
    return """
Answer to user question: {question}.
Return ONLY valid JSON array.
NO explanations. NO comments.
Each array element MUST correspond to one row.
All column names MUST BE in CamelCase (for example: CompanyName, OrderCount). No underscores allowed.
DO NOT generate any charts.
"""

@pytest.fixture(params=MODELS, ids=MODEL_IDS)
def pyai_agent_model(request):
    """PyAI agent LLM instance (parametrized)"""
    from pyai.get_model import get_model

    _logger.info(f"Running test with '{request.param[0]}:{request.param[1]}' agent")
    yield get_model(request.param[0], request.param[1])
    stop_ollamas(request.param[0])

@pytest.fixture(params=PYAI_SERVERS, ids=PYAI_SERVER_IDS)
def pyai_server(request):
    """ PyAI MCP server instance (parametrized) """
    from pydantic_ai.mcp import MCPServerStdio

    server = MCPServerStdio(
        "python",
        args=[request.param[1], "--transport", "stdio"],
        timeout=60,
    )
    yield server

@pytest.fixture
def pyai_agent(pyai_agent_model, pyai_server):
    """ PyAI agent with MCP server toolset"""
    from pydantic_ai import Agent

    agent = Agent(pyai_agent_model, toolsets=[pyai_server])

    @agent.instructions
    def mcp_server_instructions():
        return pyai_server.instructions

    yield agent