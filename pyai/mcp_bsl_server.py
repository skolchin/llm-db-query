# BSL MCP Server.
#
# Very simplistic server wrapping `create_mcp_server` from BSL package.
#
# MCP endpoint: http://127.0.0.1:7933/mcp
#
from boring_semantic_layer import from_yaml
from boring_semantic_layer.agents.backends.mcp import create_mcp_server

models = from_yaml(
    './data/northwind_bsl.yaml', 
    profile='northwind_duckdb',
    profile_path='./data/northwind_profile.yaml')

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="streamable-http")
    args = parser.parse_args()

    mcp = create_mcp_server(
        models,
        host='127.0.0.1', 
        port=7933, 
        log_level="INFO",
    )

    mcp.run(transport=args.transport)
