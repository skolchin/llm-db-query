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
import logfire
import logging
from openai import AsyncOpenAI
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.deepseek import DeepSeekProvider
from database_pydantic_ai import SQLiteDatabase,SQLDatabaseDeps, create_database_toolset
from typing import cast, Literal, Any

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
DB_FILENAME = './data/northwind.db'
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

db = SQLiteDatabase(DB_FILENAME)
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

toolset = create_database_toolset(deps=deps)
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
