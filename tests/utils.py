# Testing support functions

import re
import json
import logging
import subprocess
import pandas as pd
from typing import Dict, List

_logger = logging.getLogger(__name__)

def json_to_frame(response: str | dict | list) -> pd.DataFrame:
    """ Convert LLM response (probably JSON) to Pandas Dataframe"""

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
    """ Align expected data with LLM results considering possible discrepances, and compare them """

    # Replace underscores with dots
    result.rename(columns={col: col.replace("_", ".") for col in result.columns if '_' in col})

    # Aliases define alternative names for given result column (e.g. Categories.TotalRevenue -> TotalRevenue)
    # This provides some backlash for local LLMs
    if aliases:
        result.rename(columns={syn: col for col, synonyms in aliases.items() for syn in synonyms}, inplace=True)
        result.rename(columns={syn.lower(): col for col, synonyms in aliases.items() for syn in synonyms}, inplace=True)

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

def stop_ollamas(model_type: str | None):
    """ Stop all running ollamas """
    if not model_type or model_type == "ollama":
        subprocess.run("ollama ps | awk 'NR>1 {print $1}' | xargs -L 1 ollama stop", shell=True, check=False)
