# SQL MCP Server.
#
# An MCP server providing the tools to query the database.
#
# Basically duplicates the code from `sql_agent.py` and tools from `database-pydantic-ai`,
# but with MCP server bindings.
#
# MCP endpoint: http://127.0.0.1:7933/mcp
#
import asyncio
import logging
from dataclasses import dataclass
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from mcp.server.session import ServerSession
from mcp.server.fastmcp import FastMCP, Context
from database_pydantic_ai import SQLiteDatabase, QueryResult, SchemaInfo, TableInfo
from typing import List, Tuple, Any

logging.getLogger('database_pydantic_ai.sql.backends.sqlite').setLevel(logging.DEBUG)

# Constants
DB_FILENAME = './data/northwind.db'
""" SQLite database """

SYSTEM_PROMPT = '''
    ## SQLite Database Tools

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
    '''

@dataclass
class AppContext:
    """Application context"""
    db: SQLiteDatabase
    max_rows: int = 100
    query_timeout: float = 30.0

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Application lifecycle"""

    db = SQLiteDatabase(DB_FILENAME, echo=True)
    try:
        await db.connect()
        yield AppContext(db=db)
    finally:
        await db.close()

mcp = FastMCP(
    "SQL MCP sever",
    instructions=SYSTEM_PROMPT,
    lifespan=app_lifespan,
    host='127.0.0.1', 
    port=7933, 
    log_level="INFO",
)

@mcp.tool()
async def list_tables_tool(ctx: Context[ServerSession, AppContext]) -> List[str]:
    """
    Get names of all tables in the database to understand available data.

    Returns:
        List of all table's names.
    """
    return await ctx.request_context.lifespan_context.db.get_tables()

@mcp.tool()
async def get_schema(ctx: Context[ServerSession, AppContext], return_md: bool) -> SchemaInfo | str:
    """
    Get an overview of the database schema.

    Returns:
        List of all tables with their column counts and row counts.
    """
    return await ctx.request_context.lifespan_context.db.get_schema(return_md=return_md)

@mcp.tool()
async def describe_table(
    ctx: Context[ServerSession, AppContext], table_name: str
) -> TableInfo | str | None:
    """
    Get detailed information about a specific table.

    Args:
        table_name: Name of the table to describe.

    Returns:
        Table structure including columns, types, constraints, and relationships.
    """
    return await ctx.request_context.lifespan_context.db.get_table_info(table_name)

@mcp.tool()
async def explain_query(ctx: Context[ServerSession, AppContext], sql_query: str) -> str:
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
    return await ctx.request_context.lifespan_context.db.explain(sql_query)

@mcp.tool()
async def query(
    ctx: Context[ServerSession, AppContext], sql_query: str, max_rows: int | None = None
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
            ctx.request_context.lifespan_context.db.execute(sql_query), timeout=ctx.request_context.lifespan_context.query_timeout
        )

    except asyncio.TimeoutError:
        return QueryResult(
            columns=[],
            rows=[],
            row_count=0,
            execution_time_ms=0,  # indicate max wait with `0`
        )

    limit = max_rows or ctx.request_context.lifespan_context.max_rows

    if len(result.rows) > limit:
        result = QueryResult(
            columns=result.columns,
            rows=result.rows[:limit],
            row_count=min(result.row_count, limit),
            execution_time_ms=result.execution_time_ms,
        )

    return result

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
