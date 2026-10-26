#! /usr/bin/env python

import os
import sys

from downward.reports.absolute import AbsoluteReport

from lab.environments import BaselSlurmEnvironment, LocalEnvironment
from lab.experiment import Experiment
from lab.reports import Attribute

from parser import make_parser

from project import *

# Create custom report class with suitable info and error attributes.
class BaseReport(AbsoluteReport):
    INFO_ATTRIBUTES = ["time_limit", "memory_limit"]
    ERROR_ATTRIBUTES = [
        "domain",
        "problem",
        "algorithm",
        "unexplained_errors",
        "error",
        "node",
    ]

REMOTE = is_remote()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARKS_DIR = os.path.join(SCRIPT_DIR, "../benchmarks/ipc2023-learning/testing")
HEURISTICS = ['deepseek-r1']
DOMAINS = IPC2023_DOMAINS
TIME_LIMIT = 1800
MEMORY_LIMIT = 8192

if REMOTE:
    ENV = BaselSlurmEnvironment(email="augusto.blaascorrea@chch.ox.ac.uk",
                                partition="infai_2",
                                memory_per_cpu="5G",
                                extra_options='#SBATCH --cpus-per-task=2')
    SUITE = make_suite(DOMAINS, BENCHMARKS_DIR)
else:
    ENV = LocalEnvironment(processes=4)
    # Use smaller suite for local tests.
    SUITE = []
    for domain in DOMAINS:
        SUITE.extend(make_suite([domain], BENCHMARKS_DIR)[:1])

ATTRIBUTES = [
    "coverage",
    "error",
    "expansions",
    "initial_h_value",
    "memory",
    "plan_length",
    "run_dir",
    'score_agile',
    'score_expansion',
    "search_time",
]

exp = Experiment(environment=ENV)
exp.add_parser(make_parser())

BEST_HEURISTICS = {
    "blocksworld" : "24_10",
    "childsnack" : "19_10",
    "floortile" : "10_10",
    "miconic" : "11_10",
    "rovers" : "6_10",
    "sokoban" : "3_10",
    "spanner" : "10_10",
    "transport" : "18_10"
}

SEARCH_ALGORITHMS = [
    "gbf",
    "gbfs_early_goal_test",
    "gbfs_early_goal_test_int",
]

for domain, task in SUITE:
    for search_algorithm in SEARCH_ALGORITHMS:
        for h in HEURISTICS:
            algorithm = f"{search_algorithm}-{h}"
            heuristic_file = ''.join([domain, BEST_HEURISTICS[domain], '.py'])
            heur = os.path.join(SCRIPT_DIR, "../archive-heuristics/2025-03-10-deepseek-r1", heuristic_file)
            run = exp.add_run()
            # Create a symbolic link and an alias. This is optional. We
            # could also use absolute paths in add_command().
            run.add_resource("task", task, symlink=True)
            run.add_command(
                "plan",
                ['pypy3.11', os.path.join(SCRIPT_DIR, "../src/pyperplan/pyperplan.py"), "-H", heur, "-s", search_algorithm, task],
                time_limit=TIME_LIMIT,
                memory_limit=MEMORY_LIMIT,
            )

            task_name = os.path.basename(task)
            run.set_property("domain", domain)
            run.set_property("problem", task_name)
            run.set_property("algorithm", algorithm)
            # BaseReport needs the following properties:
            # 'time_limit', 'memory_limit', 'seed'.
            run.set_property("time_limit", TIME_LIMIT)
            run.set_property("memory_limit", MEMORY_LIMIT)
            # Every run has to have a unique id in the form of a list.
            run.set_property("id", [algorithm, domain, task_name])


# Add step that writes experiment files to disk.
exp.add_step("build", exp.build)

# Add step that executes all runs.
exp.add_step("start", exp.start_runs)

# Add step that parses the logs.
exp.add_step("parse", exp.parse)

# Add step that collects properties from run directories and
# writes them to *-eval/properties.
exp.add_fetcher(name="fetch")

# Make a report.
add_absolute_report(exp, attributes=ATTRIBUTES,
                    filter=[get_agile_score, get_expansion_score])


def filter_time_cutoff(run):
    time = run.get('search_time')
    if run['coverage'] == 1:
        if time > (19*60)+36:
        #if time > 120:
            run['coverage'] = 0
    return run

add_absolute_report(exp, attributes=ATTRIBUTES,
                    filter=[get_agile_score, get_expansion_score,filter_time_cutoff],
                    name="time-cutoff-of-e2e")

# Parse the commandline and run the given steps.
exp.run_steps()
