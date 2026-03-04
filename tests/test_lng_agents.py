# pyright: reportCallIssue=false

import re
import json
import logging
import pandas as pd
import parametrize_from_file as pff
from uuid import uuid4
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import AIMessage, ToolMessage
from typing import Dict, List

_logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.ERROR)

BSL_AGENT_PROMPT = """
Answer to user question: {question}.
Return ONLY valid JSON array.
NO explanations. NO comments.
Each array element MUST correspond to one row.
"""


def json_to_frame(response: str | dict | list) -> pd.DataFrame:

    def clear(content: str) -> str:
        # Remove invisible chars
        content = re.sub(r'[\u200b-\u200d\ufeff]', '', content)

        # Remove <think></think>
        content = re.sub(r'<think>[\s\S]*?<\/think>', '', content)

        # Extract from ```json ... ````
        if "```json" in content and not content.endswith("```"):
            content += "```"

        if (m := re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)):
            content = m.group(1)

        return content

    def clear_and_load(content: str) -> pd.DataFrame:
        if not (cleaned := clear(content)):
            raise ValueError("Empty response!")
        try:
            return pd.DataFrame(json.loads(cleaned))
        except json.JSONDecodeError as ex:
            _logger.error(f"JSON parsing error '{ex}' on content {cleaned}")
            raise

    match response:
        case str():
            return clear_and_load(response)

        case dict():
            for key in ["output", "result", "answer"]:
                if key in response:
                    return clear_and_load(response[key])
            return pd.DataFrame(response)

        case list():
            return pd.DataFrame(response)
        
        case _:
            raise ValueError(f"Don't know how to handle response of type {type(response)}")

def align_and_compare(expected: pd.DataFrame, result: pd.DataFrame, aliases: Dict[str, List[str]] | None = None):

    # Replace underscores with dots
    result.rename(columns={col: col.replace("_", ".") for col in result.columns if '_' in col})

    # Aliases define alternative names for given result column (e.g. Categories.TotalRevenue -> TotalRevenue)
    # This provides some backlash for local LLMs
    if aliases:
        result.rename(columns={syn: col for col, synonyms in aliases.items() for syn in synonyms}, inplace=True)

    # Result must have all columns from expected
    if (diff := set(expected.columns).difference(result.columns)):
        _logger.error(f"Result is missing columns {diff} (got {result.columns.tolist()})")
        raise ValueError(f"Result is missing required columns {diff}")

    # Strip extra columns from result
    if (diff := set(result.columns).difference(expected.columns)):
        _logger.warning(f"Result contains extra columns: {diff}")
        result = result[list(expected.columns)]

    # Sort frames the same way
    sort_columns = expected.select_dtypes(include=['object', 'string']).columns.tolist()
    expected = expected.sort_values(sort_columns).reset_index(drop=True)
    result = result.sort_values(sort_columns).reset_index(drop=True)

    # Round all floats to zero digits to avoid floating comparsion
    float_cols = expected.select_dtypes(include=['float']).columns
    expected[float_cols] = expected[float_cols].round(0)
    result[float_cols] = result[float_cols].round(0)

    if not (diff := expected.compare(result)).empty:
        _logger.error(f"Data difference:\n{diff}")
        raise ValueError("Result data does not match expected data")

@pff.parametrize(key="test_agents")
def test_bsl_agent(question: str, query: str, expected: str, aliases: Dict[str, List[str]], bsl_agent):
    """ Test BSL agent """

    _logger.info(f"Test question: '{question}'")

    prompt = BSL_AGENT_PROMPT.format(question=question)
    tool_output, response = bsl_agent.query(prompt)
    _logger.info(f"Response: {response}")

    expected_data = pd.DataFrame(json.loads(expected))
    response_data = json_to_frame(response)
    align_and_compare(expected_data, response_data, aliases)

def _sql_agent_trial(question: str, query: str, expected: str, aliases: Dict[str, List[str]], sql_agent):
    _logger.info(f"Test question: '{question}'")

    config = RunnableConfig({"configurable": {"thread_id": str(uuid4())}})
    response = sql_agent.invoke({"input": question}, config=config)
    _logger.info(f"Response: {response}")

    expected_data = pd.DataFrame(json.loads(expected))
    response_data = json_to_frame(response)

    align_and_compare(expected_data, response_data, aliases)

@pff.parametrize(key="test_agents")
def test_sql_agent(question: str, query: str, expected: str, aliases: Dict[str, List[str]], sql_agent):
    """ Test SQL agent """
    _sql_agent_trial(question, query, expected, aliases, sql_agent)

@pff.parametrize(key="test_agents")
def test_sql_agent_perf(question: str, query: str, expected: str, aliases: Dict[str, List[str]], sql_agent, benchmark):
    """ Benchmark SQL agent """

    status_list = []
    def safe_trial(question: str, query: str, expected: str, aliases: Dict[str, List[str]], sql_agent):
        try:
            _sql_agent_trial(question, query, expected, aliases, sql_agent)
            _logger.info("Test passed")
            status_list.append("PASSED")

        except Exception as ex:
            _logger.error(f"Error: {ex}")
            status_list.append(f"ERROR: {ex}")

    benchmark.pedantic(
        safe_trial,
        args=(question, query, expected, aliases, sql_agent),
        iterations=1,
        rounds=30,
        warmup_rounds=1)

    benchmark.extra_info['run_status'] = status_list
