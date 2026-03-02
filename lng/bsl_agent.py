# LangChain AI agent.
#
# An Agent implementing a native language queries using LangGraph / LangChain framework.
#
# Uses Boring Semantic Layer (BSL) abstraction layer. See `data/northwind_bsl.yaml`
# and /home/kol/kol/llm-db-query/data/northwind_profile.yaml` configuration files.
#
# Requires Parquet database files due to SQLite struggles with multi-threading.
#
# To run, comment `from sql_agent import app` and uncomment `from bsl_agent import app` in app.py
# and start application with streamlit:
#
# Run it with streamlit:
#
#     streamlit run lng/app.py --server.address 127.0.0.1 --server.port 7932
#
# Either local Ollama or cloud-based Deepseek / YandexGPT LLM's could be used.
# Model selection is fixed (see `MODEL_TYPE` below).
#
from pathlib import Path
from boring_semantic_layer.agents.backends.langgraph import LangGraphBackend

from get_model import get_model_name, ModelType

# Constants
MODEL_TYPE: ModelType = 'ollama'
""" Model type selection """

SYSTEM_PROMPT = '''You are a helpful database analyst assistant. Use the available tools to answer user questions about the Northwind database.'''

# Create the BSL agent
agent = LangGraphBackend(
    model_path=Path('./data/northwind_bsl.yaml'),
    llm_model=get_model_name(MODEL_TYPE),
    chart_backend="plotly",
    profile="northwind_duckdb",
    profile_file=Path("./data/northwind_profile.yaml"),
)

# Export for use in app.py
app = agent

# print(agent.query('Top 5 companies by order count'))
