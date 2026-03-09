# Generate datasets with expected query results using queriers from `tests/test_agents.yaml`.
# Will save all original test file content along with the datasets to `tests/test_agents_upd.yaml`.
#
# Uses `ruamel.yaml` because standard python_yaml does not keep comments and
# is pretty bad with long string formatting.
import os
import json
import pandas as pd
from ruamel.yaml import YAML
from sqlalchemy import create_engine
from ruamel.yaml.scalarstring import LiteralScalarString

yaml = YAML()

DB_FILENAME = './data/northwind.db'
db_file = os.path.abspath(DB_FILENAME)
engine = create_engine(f"sqlite:///{db_file}?mode=ro")

with open("tests/test_agents.yaml", "rt") as fp:
    qa_content = yaml.load(fp)

qa_key = list(qa_content.keys())[0]
qa_list = qa_content[qa_key]

qa_result_list = []
with engine.connect() as con:
    for qa_item in qa_list:
        qa_result = qa_item.copy()

        if "query" in qa_result:
            qa_result["query"] = LiteralScalarString(qa_result["query"])

        if query := qa_item.get("query"):
            data = pd.read_sql(query, con)
            qa_result["expected"] = LiteralScalarString(json.dumps(data.to_dict(orient="records"), ensure_ascii=False))
        elif "expected" in qa_result:
            qa_result["expected"] = LiteralScalarString(qa_result["expected"])

        qa_result_list.append(qa_result)

with open("tests/test_agents_upd.yaml", "wt") as fp:
    yaml.dump({qa_key: qa_result_list}, fp)
