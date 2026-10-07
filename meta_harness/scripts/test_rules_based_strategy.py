"""Feed every instruction in the tasks CSV to the router and save what it assigns.

    uv run python scripts/test_rules_based_strategy.py data/tb2.1_tasks.csv

Writes results/router_output.csv: task_id, assigned_harness, matched_rule
"""
import csv
import sys

from s.routing_strategy.rules_based import route

with open(sys.argv[1], newline="") as fin, open("results/router_output.csv", "w", newline="") as fout:
    out = csv.writer(fout)
    out.writerow(["task_id", "assigned_harness", "matched_rule"])
    for row in csv.DictReader(fin):
        harness, matched_rule = route(row["instruction"])
        out.writerow([row["task_id"], harness, matched_rule])

print("wrote results/router_output.csv")
