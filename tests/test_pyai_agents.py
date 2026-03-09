# PydanticAI agents test suite.
#
# Tests are parametrized by MCP server kind (SQL/BSL), model and QA pairs defined in `test_agents.yaml`
#
# pyright: reportCallIssue=false
import json
import logging
import pandas as pd
import parametrize_from_file as pff
from typing import Dict, List

from utils import json_to_frame, align_and_compare

_logger = logging.getLogger(__name__)

def _pyai_agent_trial(
        question: str,
        expected: str,
        aliases: Dict[str, List[str]],
        agent,
        prompt_template):
    """ Single PyAI agent trial """

    _logger.info(f"Test question: '{question}'")

    prompt = prompt_template.format(question=question)
    response = agent.run_sync(prompt)
    _logger.info(f"Response: {response.output}")

    expected_data = pd.DataFrame(json.loads(expected))
    response_data = json_to_frame(response.output)
    align_and_compare(expected_data, response_data, aliases)

@pff.parametrize(path="test_agents.yaml", key="test_agents")
def test_pyai_agent(
    question: str,
    query: str,
    expected: str,
    aliases: Dict[str, List[str]],
    pyai_agent,
    pyai_agent_prompt):
    """ Test PyAI agent """

    _pyai_agent_trial(question, expected, aliases, pyai_agent, pyai_agent_prompt)

@pff.parametrize(path="test_agents.yaml", key="test_agents")
def test_pyai_agent_perf(
    question: str,
    query: str,
    expected: str,
    aliases: Dict[str, List[str]],
    pyai_agent,
    pyai_agent_prompt,
    benchmark,
    num_perf_rounds):
    """ Benchmark PyAI agent """

    status_list = []
    def safe_trial():
        try:
            _pyai_agent_trial(question, expected, aliases, pyai_agent, pyai_agent_prompt)
            _logger.info("Test passed")
            status_list.append("PASSED")

        except Exception as ex:
            _logger.error(f"Error: {ex}")
            status_list.append(f"ERROR: {ex}")

    benchmark.pedantic(
        safe_trial,
        iterations=1,
        rounds=num_perf_rounds,
        warmup_rounds=1)

    # status list includes warmup round
    benchmark.extra_info['run_status'] = status_list[1:]
