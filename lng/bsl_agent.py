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
# Either local Ollama or cloud Deepseek LLM's could be used. YandexGPT is not supported!
# Model selection is fixed (see `MODEL_TYPE` below).
#
from pathlib import Path
from boring_semantic_layer.agents.backends.langgraph import LangGraphBackend

from get_model import get_model_qualified_name, ModelType

# Constants
MODEL_TYPE: ModelType = "deepseek"
""" Model type selection """

SYSTEM_PROMPT = '''You are a helpful database analyst assistant. Use the available tools to answer user questions about the Northwind database.'''

# Create the BSL agent
agent = LangGraphBackend(
    model_path=Path('./data/northwind_bsl.yaml'),
    llm_model=get_model_qualified_name(MODEL_TYPE, "deepseek-reasoner"),
    chart_backend="plotly",
    profile="northwind_duckdb",
    profile_file=Path("./data/northwind_profile.yaml"),
    return_json=True,
)

# Export for use in app.py
app = agent

# Testing
if __name__ == '__main__':
    import logging
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--query", default="Top 5 companies by order count")
    parser.add_argument("--debug", action="store_true")
    args = vars(parser.parse_args())

    logging.basicConfig(
        format='[%(levelname).1s %(asctime)s %(name)s] %(message)s',
        level=logging.INFO if not args["debug"] else logging.DEBUG,
        force=True)

    def on_thinking(t):
        if args["debug"]:
            print(f'\tthinking: {t}')

    def on_tool_call(name, tool_args, tokens):
        if args["debug"]:
            print(f'\ttool_call: {name} ({tool_args})')

    def on_error(e):
        if args["debug"]:
            print(f'\terror: {e}')

    def on_tool_result(tool_call_id, status, error, content):
        if args["debug"]:
            print(f'\ttool_result: {tool_call_id} ({status})')

    query = args["query"]
    tool_output, response = agent.query(query,
                                        on_thinking=on_thinking,
                                        on_tool_call=on_tool_call,
                                        on_error=on_error,
                                        on_tool_result=on_tool_result)
    
    print(f"=== Final response:\n{response}\n\n=== Final tool output:\n{tool_output}")
