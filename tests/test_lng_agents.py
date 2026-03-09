# LangChain / LangGraph agents test suite
#
# Tests are parametrized by model and QA pairs defined in `test_agents.yaml`.
#
# pyright: reportCallIssue=false
import json
import logging
import pandas as pd
import parametrize_from_file as pff
from uuid import uuid4
from langchain_core.runnables.config import RunnableConfig
from typing import Dict, List

from utils import json_to_frame, align_and_compare

_logger = logging.getLogger(__name__)

def _lng_bsl_agent_trial(
        question: str,
        expected: str,
        aliases: Dict[str, List[str]],
        agent,
        prompt_template):
    """ Single BSL agent trial """
    _logger.info(f"Test question: '{question}'")

    prompt = prompt_template.format(question=question)
    tool_output, response = agent.query(prompt)
    _logger.info(f"Response: {response}")

    expected_data = pd.DataFrame(json.loads(expected))
    response_data = json_to_frame(response)
    align_and_compare(expected_data, response_data, aliases)

def _lng_sql_agent_trial(
        question: str, 
        expected: str, 
        aliases: Dict[str, List[str]], 
        agent):
    """ Single SQL agent trial """
    _logger.info(f"Test question: '{question}'")

    config = RunnableConfig({"configurable": {"thread_id": str(uuid4())}})
    response = agent.invoke({"input": question}, config=config)
    _logger.info(f"Response: {response}")

    expected_data = pd.DataFrame(json.loads(expected))
    response_data = json_to_frame(response)

    align_and_compare(expected_data, response_data, aliases)

@pff.parametrize(path="test_agents.yaml", key="test_agents")
def test_lng_bsl_agent(question: str, query: str, expected: str, aliases: Dict[str, List[str]], lng_bsl_agent, lng_bsl_agent_prompt):
    """ Test BSL agent """
    _lng_bsl_agent_trial(question, expected, aliases, lng_bsl_agent, lng_bsl_agent_prompt)

@pff.parametrize(path="test_agents.yaml", key="test_agents")
def test_lng_sql_agent(question: str, query: str, expected: str, aliases: Dict[str, List[str]], lng_sql_agent):
    """ Test SQL agent """
    _lng_sql_agent_trial(question, expected, aliases, lng_sql_agent)

@pff.parametrize(path="test_agents.yaml", key="test_agents")
def test_lng_bsl_agent_perf(
    question: str,
    query: str,
    expected: str,
    aliases: Dict[str, List[str]],
    lng_bsl_agent,
    lng_bsl_agent_prompt,
    benchmark,
    num_perf_rounds):
    """ Benchmark BSL agent """

    status_list = []
    def safe_trial():
        try:
            _lng_bsl_agent_trial(question, expected, aliases, lng_bsl_agent, lng_bsl_agent_prompt)
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

@pff.parametrize(path="test_agents.yaml", key="test_agents")
def test_lng_sql_agent_perf(
    question: str,
    query: str,
    expected: str,
    aliases: Dict[str, List[str]],
    lng_sql_agent,
    benchmark,
    num_perf_rounds):
    """ Benchmark SQL agent """

    status_list = []
    def safe_trial():
        try:
            _lng_sql_agent_trial(question, expected, aliases, lng_sql_agent)
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
