# Pydantic AI agent.
#
# An agent implementing a native language queries using Pydantic AI framework.
#
# Run as ordinary Python app or with uvicorn command:
#
#     uvicorn pyai.sql_agent:app --host 127.0.0.1 --port 7932
#
import os
import yaml
import asyncio
import logging
from database_pydantic_ai import SQLiteDatabase, SQLDatabaseDeps, create_database_toolset
from pydantic_ai import Agent, ModelRetry, RunContext
from typing import cast

from pyai.get_model import get_models

# Logging configuration
logging.basicConfig(
    format='[%(levelname).1s %(asctime)s %(name)s] %(message)s',
    level=logging.INFO,
    force=True)

logger = logging.getLogger(__name__)
logging.getLogger('database_pydantic_ai.sql.backends.sqlite').setLevel(logging.DEBUG)

# Constants
DB_FILENAME = './data/northwind.db'
""" SQLite database """

# Database setup
db = SQLiteDatabase(DB_FILENAME, echo=True)
""" Database instance """

deps = SQLDatabaseDeps(database=db)
""" Global dependency for database """

SYSTEM_PROMPT = '''
## SQLite Database Toolset - Instructions

You are an database analyst with access to SQLite database tools.
Your task is to answer to user questions by retrieving data from database while strictly adhering to the rules and procedures outlined below.

### Available tools:

* `list_tables()` - returns a list of all tables in the database.
* `get_schema()` - returns the full database schema (all tables, indexes, foreign keys).
* `describe_table(table_name: str)` - returns detailed structure of the specified table: columns, data types, NULL/NOT NULL, primary key, foreign keys, indexes.
* `explain_query(sql: str)` - shows the query execution plan and dependencies.
* `query(sql: str)` - executes an SQL query (SELECT only) and returns the data.

### Constraints
* **SELECT only**. No INSERT, UPDATE, DELETE, CREATE, ALTER, DROP.
* **SQLite functions only** - use only functions supported by SQLite.
* **Always validate the schema** before constructing queries with JOIN.
* **Never guess** about relationships between tables, rely solely on schema information.
* Do not ask for permission to execute a query, run it immediately (if you are confident it is correct).

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
""" The system prompt """

toolset = create_database_toolset()
""" The toolset """

models = get_models()
""" Available models """

agent = Agent(
    models['ollama'],
    toolsets=[toolset],
    deps_type=SQLDatabaseDeps,
    builtin_tools=[],
    instructions=SYSTEM_PROMPT,
)
""" The agent instance """

# Meta tool
# OverviewKind = Literal['title', 'origin', 'description', 'usage', 'structure']
# @agent.tool
# def get_database_overview(ctx: RunContext[SQLDatabaseDeps], kind: OverviewKind) -> str | None:
#     """
#     Retrieve a concise, high‑level overview of the database.
#     This tool is the first‑stop for anyone who wants to understand what the database is, why it exists, 
#     and how it’s structured without diving into raw table definitions.

#     Args:
#         kind: Overview section kind, one of: 'title', 'origin', 'description', 'usage', 'structure'

#     Returns:
#         Overview section content
#     """
#     if not isinstance(ctx.deps.database, SQLiteDatabase):
#         return None
    
#     meta_file, _ = os.path.splitext(cast(SQLiteDatabase, ctx.deps.database).db_path)
#     meta_file += '.yaml'
#     try:
#         with open(meta_file, 'rt') as fp:
#             meta_dict = yaml.safe_load(fp)

#             if not 'database' in meta_dict:
#                 return None

#             if (value := meta_dict['database'].get(kind)) is None:
#                 raise ModelRetry(f'Cannot find {kind} key in {meta_file}. Use another overview kind and try again.')
            
#             return str(value)
        
#     except FileNotFoundError:
#         return None

app = agent.to_web(models=models, deps=deps)
""" Starlette UI application """

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=7932, log_level='info')
