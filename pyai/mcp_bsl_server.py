# BSL MCP Server.
#
# Very simplistic server wrapping `create_mcp_server` from BSL package.
#
# MCP endpoint: http://127.0.0.1:7933/mcp
#
# Use this config for OpenClaw / Claude (replace {ROOT} with absolute path):
#
# {
#  "mcpServers": {
#    "northwind": {
#      "command": "{ROOT}/.venv/bin/python",
#      "args": ["{ROOT}/pyai/mcp_bsl_server.py", "--transport", "stdio", "--fix-path"],
#      "cwd": "{ROOT}"
#    }
#  },
#  "imports": []
#}

from pathlib import Path
from argparse import ArgumentParser
from boring_semantic_layer import from_yaml
from boring_semantic_layer.agents.backends.mcp import create_mcp_server

from fix_path import fix_data_paths

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data"

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="streamable-http")
    parser.add_argument("--fix-path", action="store_true")
    args = parser.parse_args()

    profile_path = DATA_ROOT / "northwind_profile.yaml"
    config_path = DATA_ROOT / "northwind_bsl.yaml"

    if args.fix_path:
        # This will replace relative data file paths set in BSL profile to absolute ones
        # allowing to start the server from arbitrary directory.
        profile_path = fix_data_paths(profile_path, ROOT)

    models = from_yaml(
        str(config_path),
        profile_path=str(profile_path))

    mcp = create_mcp_server(
        models,
        host='127.0.0.1', 
        port=7933, 
        log_level="INFO",
    )

    mcp.run(transport=args.transport)
