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