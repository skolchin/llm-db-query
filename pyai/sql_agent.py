# Pydantic AI agent.
#
# An Agent implementing a native language queries using Pydantic AI framework.
#
# Run as ordinary Python app or with uvicorn command:
#
#     uvicorn pyai.sql_agent:app --host 127.0.0.1 --port 7932
#
# Either local Ollama or cloud-based Deepseek / YandexGPT LLM's could be used.
# Models are loaded upon startup (if credentials set) and can be selected from UI.
#

import os
import dotenv
import asyncio
import logfire
import logging
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai import Agent, FunctionToolset, RunContext, ModelSettings
from database_pydantic_ai import SQLiteDatabase,SQLDatabaseDeps
from database_pydantic_ai.types import QueryResult, SchemaInfo, TableInfo
from typing import Literal, cast

ModelType = Literal['ollama', 'deepseek', 'yandex']
""" Allowed model types  """

# Load environment from .env
dotenv.load_dotenv()

# Logging configuration
logging.basicConfig(
    format='[%(levelname).1s %(asctime)s %(name)s] %(message)s',
    level=logging.INFO,
    force=True)

logfire.configure(
    send_to_logfire=False,
    console=logfire.ConsoleOptions(min_log_level='info', verbose=True)
)

logger = logging.getLogger(__name__)

# Constants
DB_FILENAME = './data/titanic.db'
""" SQLite database """

MODEL_SETTINGS = ModelSettings(
    temperature=0.1,
    timeout=60,
)

# Setup the model
def get_model(model_type: ModelType) -> OpenAIChatModel | None:
    """ Return Agent LLM instance """

    match model_type:
        case 'ollama':
            # Local Ollama instance provider
            return OpenAIChatModel(
                model_name=os.environ.get('OLLAMA_MODEL', 'gpt-oss:20b'),
                provider= OllamaProvider('http://localhost:11434/v1'),
            )

        case 'deepseek':
            # Deepseek provider
            if not 'DEEPSEEK_API_KEY' in os.environ:
                return None
            
            return OpenAIChatModel(
                model_name=os.environ.get('DEEPSEEK_MODEL', 'deepseek-chat'),
                provider=DeepSeekProvider(api_key=os.environ['DEEPSEEK_API_KEY']),
            )

        case 'yandex':
            # OpenAI-compatible YandexGPT provider
            if not 'YANDEX_API_KEY' in os.environ or not 'YANDEX_FOLDER_ID' in os.environ:
                return None

            client = AsyncOpenAI(
                api_key=os.environ['YANDEX_API_KEY'],
                base_url='https://ai.api.cloud.yandex.net/v1',
                project=os.environ['YANDEX_FOLDER_ID'],
            )
            return OpenAIChatModel(
                model_name=f"gpt://{os.environ['YANDEX_FOLDER_ID']}/{os.environ.get('YANDEX_MODEL', 'yandexgpt')}/latest",
                provider=OpenAIProvider(openai_client=client),
            )

        case _:
            raise ValueError(f'Unknown model type {model_type}')

all_models = {k: get_model(cast(ModelType,k)) for k in ['ollama', 'deepseek', 'yandex']}
models = {k: m for k, m in all_models.items() if m is not None}
""" LLM models available to agent """

class SQLiteDatabaseExt(SQLiteDatabase):
    """ Subclassed `SQLiteDatabase` class from `database_pydantic_ai`.

    Establishes connection before executing statements and logs all SQL statements.
    """
    async def execute(self, query, params = None):
        if self._connection is None:
            await self.connect()
            assert self._connection is not None
            await self._connection.set_trace_callback(logger.info) # pyright: ignore[reportGeneralTypeIssues]
        return await super().execute(query, params)

db = SQLiteDatabaseExt(DB_FILENAME)
""" Database instance """

deps = SQLDatabaseDeps(database=db)
""" Global dependency for database """

SYSTEM_PROMPT = '''
## SQLite Database Toolset - Instructions

You are an database analyst with access to SQLite database tools.
Your task is to answer to user questions by retrieving data from database while strictly adhering to the rules and procedures outlined below.

### Available tools for database operations and querying:

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
""" The system prompt """

def create_database_toolset_ext(*, id: str | None = None) -> FunctionToolset[SQLDatabaseDeps]:
    """
    Create a database toolset for AI Agents.

    This is a replica of `database_pydantic_ai.create_database_toolset` function,
    which handles 'dependency missing' problem for web-based agents by using global dependency.

    Tool definitions and functionality are the same as original ones.
    """
    toolset = FunctionToolset[SQLDatabaseDeps](id=id)

    @toolset.tool
    async def list_tables(ctx: RunContext[SQLDatabaseDeps]) -> list[str]:
        """
        Get names of all tables in the database to understand available data.

        Returns:
            List of all table's names.
        """
        return await (ctx.deps or deps).database.get_tables()

    @toolset.tool
    async def get_schema(ctx: RunContext[SQLDatabaseDeps], return_md: bool) -> SchemaInfo | str:
        """
        Get an overview of the database schema.

        Returns:
            List of all tables with their column counts and row counts.
        """
        return await (ctx.deps or deps).database.get_schema(return_md=return_md)

    @toolset.tool
    async def describe_table(
        ctx: RunContext[SQLDatabaseDeps], table_name: str
    ) -> TableInfo | str | None:
        """
        Get detailed information about a specific table.

        Args:
            table_name: Name of the table to describe.

        Returns:
            Table structure including columns, types, constraints, and foreign keys.
        """
        return await (ctx.deps or deps).database.get_table_info(table_name)

    @toolset.tool
    async def explain_query(ctx: RunContext[SQLDatabaseDeps], sql_query: str) -> str:
        """
        Get the execution plan for a SQL query without executing it.

        Args:
            sql_query: The SQL query to analyze.

        Returns:
            Query execution plan showing how the database would process the query.

        Use this to:
            - Understand query performance
            - Identify missing indexes
            - Optimize slow queries
        """
        return await (ctx.deps or deps).database.explain(sql_query)

    @toolset.tool
    async def query(
        ctx: RunContext[SQLDatabaseDeps], sql_query: str, max_rows: int | None = None
    ) -> QueryResult:
        """
        Execute a SQL query and return the results.

        Args:
            sql_query: SQL query to be executed.
            max_rows: Maximum number of rows to be returned (default: 100)

        Returns:
            QueryResults object with queried data.

        Example:
            query("SELECT id, name FROM users WHERE is_banned = true;", max_rows=10)
        """
        try:
            result = await asyncio.wait_for(
                (ctx.deps or deps).database.execute(sql_query), timeout=(ctx.deps or deps).query_timeout
            )

        except asyncio.TimeoutError:
            return QueryResult(
                columns=[],
                rows=[],
                row_count=0,
                execution_time_ms=0,  # indicate max wait with `0`
            )

        limit = max_rows or (ctx.deps or deps).max_rows

        if len(result.rows) > limit:
            result = QueryResult(
                columns=result.columns,
                rows=result.rows[:limit],
                row_count=min(result.row_count, limit),
                execution_time_ms=result.execution_time_ms,
            )

        return result

    return toolset

toolset = create_database_toolset_ext()
""" The toolset """

agent = Agent(
    models['ollama'],
    toolsets=[toolset],
    deps_type=SQLDatabaseDeps,
    system_prompt=SYSTEM_PROMPT,
    model_settings=MODEL_SETTINGS,
)
""" The agent instance """

app = agent.to_web(deps=deps, models=models)
""" Starlette UI application. `deps` reference provided here is ignored (probably a bug) """

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=7932, log_level='info')
