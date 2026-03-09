# Pydantic AI agent with MCP support.
#
# An agent implementing a native language queries using Pydantic AI framework
# and SQL or BSL MCP server.
#
# Run as normal Python app or with uvicorn:
#
#     uvicorn pyai.mcp_agent:app --host 127.0.0.1 --port 7932
#
# To start MCP server run either
#
#   python pyai/mcp_sql_server.py
#
# or
#
#   python pyai/mcp_bsl_server.py
#
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from get_model import get_models

models = get_models()
server = MCPServerStreamableHTTP('http://127.0.0.1:7933/mcp')
agent = Agent(models['ollama'], toolsets=[server])  

@agent.instructions
async def mcp_server_instructions():
    return server.instructions 

app = agent.to_web(models=models)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=7932, log_level='info')
