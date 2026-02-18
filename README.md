# Natural Language Query Agents

This project is a playground for experimenting with AI agents capable of handling **natural language queries** to retrieve and analyze data from underlying SQL databases.

## Overview

The repository features agents built with different frameworks:
* **LangChain Agent** (`lng/`)
* **PydanticAI Agent** (`pyai/`)

These agents use LLMs to translate natural language into SQL `SELECT` statements, execute them against raw database tables, and transform the results into a human-readable answer. The agents are capable of handling sophisticated queries involving multi-table joins and complex groupings.

### Example Queries
* *"Analyze the survival rate across all available factors and summarize the most significant ones."* (Titanic DB)
* *"Compare seafood ordering across all available years."* (Northwind DB)
* *"Provide monthly usage for the summer of 2017 for all categories."* (Northwind DB)

All database operations are encapsulated in **tools**. Agents use these tools to discover the database structure, inspect table schemas, and manage joins. While effective for simple to moderately complex databases, reliability may vary with highly intricate schemas.

---

## Model Support

The project supports any LLM compatible with the **OpenAI Completions API**. Tested providers include:
* **[Ollama](https://ollama.com)** (Local)
* **[DeepSeek](https://deepseek.com)**
* **[YandexGPT](https://yandex.cloud/en/services/yandexgpt)**

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
1. **titanic.db**: The [Titanic dataset](https://www.kaggle.com).
2. **northwind.db**: The classic Microsoft [Northwind sample database](https://github.com) with simulated sales data.

To switch databases, update the `DB_FILENAME` constant in the main agent file.

---

## Setup & Usage

### 1. Environment
Create a **Python 3.12+** environment and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Ollama Installation (Optional)
If running locally on Linux:
```bash
curl -fsSL https://ollama.com | sh
ollama pull <MODEL_NAME>
```

### 3. Running the Agents
Refer to the specific agent files (e.g., xxx_agent.py) for execution instructions.
Each agent launches a simple chatbot interface accessible at [http://127.0.0.1:7932](http://127.0.0.1:7932).
