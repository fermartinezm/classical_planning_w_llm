#! /usr/bin/env python

import glob
import os
import sys

from random import shuffle

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


REMOTE = BaselSlurmEnvironment.is_present()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if REMOTE:
    BENCHMARKS_DIR = "/infai/blaas/projects/beating-learning-ipc/benchmarks/hidden/testing"
else:
    BENCHMARKS_DIR = "/home/blaas/work/projects/llm-related-projects/beating-learning-ipc/benchmarks/hidden/testing"

DOMAINS = ["blocksworld"]
HEURISTICS = ['hff', 'hadd']
TIME_LIMIT = 1800
MEMORY_LIMIT = 8192

if REMOTE:
    ENV = BaselSlurmEnvironment(email="augusto.blaascorrea@chch.ox.ac.uk")
    SUITE = make_suite(DOMAINS, BENCHMARKS_DIR)
else:
    ENV = LocalEnvironment(processes=4)
    # Use smaller suite for local tests.
    SUITE = [make_suite(DOMAINS, BENCHMARKS_DIR)[0]]

ATTRIBUTES = [
    "plan_length",
    "search_time",
    "expansions",
    "error",
    "initial_h_value",
    "coverage",
]



# Create a new experiment.
exp = Experiment(environment=ENV)
# Add solver to experiment and make it available to all runs.
#exp.add_resource("solver", os.path.join(SCRIPT_DIR, "solver.py"))
# Add custom parser.
exp.add_parser(make_parser())

for domain, task in SUITE:
    for h in HEURISTICS:
        algorithm = h
        run = exp.add_run()
        # Create a symbolic link and an alias. This is optional. We
        # could also use absolute paths in add_command().
        run.add_resource("task", task, symlink=True)
        run.add_command(
            "plan",
            [sys.executable, os.path.join(SCRIPT_DIR, "../src/pyperplan/pyperplan.py"), "-H", h, "-s", "gbf", task],
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
exp.add_report(BaseReport(attributes=ATTRIBUTES), outfile="report.html")

# Parse the commandline and run the given steps.
exp.run_steps()
