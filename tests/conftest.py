import os
import pytest
import logging
import subprocess
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from lng.get_model import get_model_qualified_name, get_model

_logger = logging.getLogger(__name__)

# ("deepseek", "deepseek-reasoner") removed as it's not good with tool calls
# see https://github.com/langchain-ai/langchain/issues/34166
MODELS = [("ollama", "gpt-oss:20b"),
          ("ollama", "qwen3:30b"),
          ("deepseek", "deepseek-chat"),]

MODEL_IDS = [v[1] for v in MODELS]

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
def agent_model(request):
    """Agent LLM instance (parametrized)"""
    _logger.info(f"Running test with '{request.param[0]}:{request.param[1]}' agent")
    try:
        yield get_model(request.param[0], request.param[1])
    finally:
        # Stop all running ollamas
        if request.param[0] == "ollama":
            subprocess.run("ollama ps | awk 'NR>1 {print $1}' | xargs -L 1 ollama stop", shell=True, check=False)

@pytest.fixture(params=MODELS, ids=MODEL_IDS)
def agent_model_qual_name(request):
    """Agent LLM qualified name (parametrized)"""
    _logger.info(f"Running test with '{request.param[0]}:{request.param[1]}' agent")

    yield get_model_qualified_name(request.param[0], request.param[1])

    # Stop all running ollamas
    if request.param[0] == "ollama":
        subprocess.run("ollama ps | awk 'NR>1 {print $1}' | xargs -L 1 ollama stop", shell=True, check=False)

@pytest.fixture(scope='session')
def bsl_agent_prompt() -> str:
    return """
Answer to user question: {question}.
Return ONLY valid JSON array.
NO explanations. NO comments.
Each array element MUST correspond to one row.
All column names MUST BE in CamelCase (for example CompanyName, OrderCount). No underscores allowed.
"""

@pytest.fixture(scope='session')
def sql_agent_prompt() -> str:
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
def bsl_agent(agent_model_qual_name):
    """ BSL agent instance """
    from boring_semantic_layer.agents.backends.langgraph import LangGraphBackend

    agent = LangGraphBackend(
        model_path=Path('./data/northwind_bsl.yaml'),
        llm_model=agent_model_qual_name,
        chart_backend="plotext",
        profile="northwind_duckdb",
        profile_file=Path("./data/northwind_profile.yaml"),
        return_json=True,
    )
    yield agent

@pytest.fixture
def sql_agent(agent_model, sqldb_database, sql_agent_prompt):
    from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
    from langchain_community.agent_toolkits.sql.base import create_sql_agent

    toolkit = SQLDatabaseToolkit(db=sqldb_database, llm=agent_model)

    prompt = ChatPromptTemplate.from_messages([
        ("system", sql_agent_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_sql_agent(
        top_k=1000,
        prompt=prompt,
        llm=agent_model,
        toolkit=toolkit,
        agent_type="tool-calling",
        verbose=True,
    )
    return agent
