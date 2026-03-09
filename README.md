# Natural Language Query Agents

This project is a playground for experimenting with AI agents capable of handling **natural language queries** to retrieve and analyze data from underlying SQL databases.

## Overview

The repository features agents built with different frameworks:
- **LangChain Agent** (`lng/`)
- **PydanticAI Agent** (`pyai/`)

These agents use LLMs to translate natural language either into SQL `SELECT` statements (executed directly against database tables) or into queries against a middleware semantic layer (see [Boring Semantic Layer](https://boringdata.github.io/boring-semantic-layer/)), which are themselves translated into SQL queries. In both cases, the selected data is then transformed back into a human-readable answer by the LLMs.

The agents are capable of handling sophisticated queries involving multi-table joins and complex groupings.

### Example Queries

- *“Analyze the survival rate across all available factors and summarize the most significant ones.”* (Titanic DB)
- *“Compare seafood ordering across all available years.”* (Northwind DB)
- *“Provide monthly usage for the summer of 2016 for all categories.”* (Northwind DB)

All database operations are encapsulated in **tools**. Agents use these tools to discover the database structure, inspect table schemas, and manage joins. While effective for simple to moderately complex databases, reliability may vary with highly intricate schemas.

The tools can be attached directly to the agent or implemented as a separate MCP server.

---

## Model Support

The project supports any LLM compatible with the **OpenAI Completions API**. Tested providers and models include:
- **[Ollama](https://ollama.com)**: `gpt-oss`, `qwen3`, `nemotron-3-nano`
- **[DeepSeek](https://deepseek.com)**: `deepseek-chat`
- **[Yandex](https://yandex.cloud/en/services/yandexgpt)**: `yandexgpt`.

More providers and models are to come.

> [!IMPORTANT]  
> When using Ollama, ensure you use a model that supports **Function Calling/Tools**. The `gpt-oss:20b` (OpenAI’s open-weighted GPT) is recommended for its reliability in tool usage.

### Configuration
Set your parameters in the `.env` file (create one from `.env.example`):

| Provider | Variables |
| :--- | :--- |
| **Ollama** | `OLLAMA_MODEL` (default: `gpt-oss:20b`) |
| **DeepSeek** | `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL` (default: `deepseek-chat`) |
| **Yandex** | `YANDEX_API_KEY`, `YANDEX_FOLDER_ID`, `YANDEX_MODEL` (default: `yandexgpt`) |

---

## Datasets

The `./data` folder contains two SQLite databases:
1. **titanic.db**: The [Titanic dataset](https://www.kaggle.com/c/titanic) from Kaggle.
2. **northwind.db**: The classic Microsoft [Northwind sample database](https://github.com/microsoft/sql-server-samples/tree/master/samples/databases/northwind-pubs) (adapted for SQLite).

To switch databases, update the `DB_FILENAME` constant in the main agent file.

---

## Setup & Usage

### 1. Environment

Create a **Python 3.12+** virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Ollama Installation (Optional)

To install Ollama locally on Linux, run:

```bash
curl -fsSL https://ollama.com | sh
ollama pull <MODEL_NAME>
```

### 3. Running the Agents

Refer to the specific agent files (e.g., `sql_agent.py` or `mcp_agent.py`) for execution instructions. 
Each agent launches a simple chatbot interface accessible at http://127.0.0.1:7932.

## Testing

Test suites are available in the `tests` directory, along with a separate `requirements.txt` file.

Individual test names indicate what is being tested: `lng` / `pyai` (LangChain or PydanticAI agents) and `sql` / `bsl` (direct SQL or BSL agents). Each test queries the agent on a particular question, collects the output, and compares the data with expected results. Test QA pairs are available in the `tests/test_agents.yaml` file.

In addition to functionality testing, there are performance tests implemented with the `pytest-benchmark` plugin. These tests run each question for a predefined number of times (set by the `benchmark-min-rounds` option) and save execution times to a JSON file under the `.benchmarks` directory. Use the `tests/support/view_benchmarks.ipynb` notebook to view and compare the results.
